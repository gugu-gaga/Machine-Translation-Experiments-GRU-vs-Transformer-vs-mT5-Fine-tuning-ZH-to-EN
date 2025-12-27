#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, sys, time, subprocess, json
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from utils.common import set_seed, now_ts, ensure_dir, JsonlLogger, save_json
from utils.config import load_yaml, save_yaml
from utils.io import load_ids, load_text, save_lines
from utils.dist import init_distributed, ddp_env, is_main_process, barrier
from utils.batching import make_token_batches, shard_batches
from models.rnn_nmt import RNNAttnSeq2Seq, load_pretrained_embeddings

def get_tf_ratio(cfg: Dict[str, Any], step: int) -> float:
    sch = cfg["train"]["tf_schedule"]
    if sch["type"] == "constant":
        return float(sch["ratio"])
    if sch["type"] == "warmup_then_constant":
        return 1.0 if step < int(sch["warmup_steps"]) else float(sch["ratio_after"])
    raise ValueError(sch)

def collate(indices, src_ids, tgt_ids, pad_id, bos_id, eos_id, device):
    src_seqs=[src_ids[i]+[eos_id] for i in indices]
    tgt_seqs=[tgt_ids[i] for i in indices]
    B=len(indices)
    S=max(len(x) for x in src_seqs)
    T=max(len(x) for x in tgt_seqs)+1
    src=torch.full((B,S),pad_id,dtype=torch.long)
    src_mask=torch.zeros((B,S),dtype=torch.bool)
    tgt_in=torch.full((B,T),pad_id,dtype=torch.long)
    tgt_out=torch.full((B,T),pad_id,dtype=torch.long)
    for b,(s,t) in enumerate(zip(src_seqs,tgt_seqs)):
        src[b,:len(s)]=torch.tensor(s,dtype=torch.long)
        src_mask[b,:len(s)]=True
        tgt_in[b,0]=bos_id
        if len(t)>0:
            tgt_in[b,1:1+len(t)]=torch.tensor(t,dtype=torch.long)
            tgt_out[b,:len(t)]=torch.tensor(t,dtype=torch.long)
        tgt_out[b,len(t)]=eos_id
    return src.to(device), src_mask.to(device), tgt_in.to(device), tgt_out.to(device)

@torch.no_grad()
def greedy_decode(model, sp, src_ids, pad_id, bos_id, eos_id, max_new_tokens, device, batch_size=64):
    model.eval()
    hyps=[]
    for st in range(0,len(src_ids),batch_size):
        batch=src_ids[st:st+batch_size]
        src_seqs=[x+[eos_id] for x in batch]
        B=len(src_seqs); S=max(len(x) for x in src_seqs)
        src=torch.full((B,S),pad_id,dtype=torch.long,device=device)
        mask=torch.zeros((B,S),dtype=torch.bool,device=device)
        for i,s in enumerate(src_seqs):
            src[i,:len(s)]=torch.tensor(s,dtype=torch.long,device=device)
            mask[i,:len(s)]=True
        enc_out, enc_h = model.encoder(src, mask)
        state=enc_h
        prev=torch.full((B,),bos_id,dtype=torch.long,device=device)
        finished=torch.zeros((B,),dtype=torch.bool,device=device)
        out_ids=[[] for _ in range(B)]
        for _ in range(max_new_tokens):
            logits,state,_=model.decoder.forward_step(prev,state,enc_out,mask)
            nxt=torch.argmax(logits,dim=-1)
            prev=nxt
            for i in range(B):
                if finished[i]: continue
                tid=int(nxt[i].item())
                if tid==eos_id: finished[i]=True
                else: out_ids[i].append(tid)
            if finished.all(): break
        for i in range(B):
            hyps.append(sp.decode(out_ids[i]))
    return hyps

def save_ckpt(path, model, opt, scaler, step, best_bleu, cfg):
    torch.save({"model":model.state_dict(),"opt":opt.state_dict(),"scaler":scaler.state_dict(),
                "step":int(step),"best_bleu":float(best_bleu),"cfg":cfg}, path)

def run_eval(cfg, run_dir, base_model, device, step):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=cfg["data"]["spm_model"])

    valid_src_ids = load_ids(cfg["data"]["valid_src_ids"])
    valid_tgt_ids = load_ids(cfg["data"]["valid_tgt_ids"])

    # -------- 1) teacher-forcing valid CE loss / PPL --------
    base_model.eval()
    pad_id = int(cfg["data"]["pad_id"])
    bos_id = int(cfg["data"]["bos_id"])
    eos_id = int(cfg["data"]["eos_id"])

    loss_sum_fn = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    total_nll = 0.0
    total_tok = 0

    bs = int(cfg["decode"]["eval_batch_size"])
    with torch.no_grad():
        for st in range(0, len(valid_src_ids), bs):
            idxs = list(range(st, min(st + bs, len(valid_src_ids))))
            # 用已有 collate（teacher forcing：tgt_in 全是 gold 前缀）
            src, src_mask, tgt_in, tgt_out = collate(
                idxs,
                valid_src_ids,
                valid_tgt_ids,
                pad_id, bos_id, eos_id,
                device
            )
            enc_out, enc_h = base_model.encoder(src, src_mask)
            state = enc_h

            B, T = tgt_in.shape
            prev = tgt_in[:, 0]
            for t in range(T):
                logits, state, _ = base_model.decoder.forward_step(prev, state, enc_out, src_mask)
                gold = tgt_out[:, t]
                total_nll += float(loss_sum_fn(logits, gold).item())
                total_tok += int((gold != pad_id).sum().item())
                if t + 1 < T:
                    prev = tgt_in[:, t + 1]

    valid_loss_ce = total_nll / max(total_tok, 1)
    valid_ppl = float(torch.exp(torch.tensor(valid_loss_ce)).item())

    # -------- 2) greedy decode + BLEU --------
    valid_src_txt = load_text(cfg["data"]["valid_src_txt"])
    valid_ref = load_text(cfg["data"]["valid_tgt_txt"])

    t0 = time.time()
    hyps = greedy_decode(
        base_model, sp, valid_src_ids,
        pad_id, bos_id, eos_id,
        max_new_tokens=int(cfg["decode"]["max_new_tokens_eval"]),
        device=device,
        batch_size=int(cfg["decode"]["eval_batch_size"])
    )
    dt = time.time() - t0

    decode_dir = ensure_dir(os.path.join(run_dir, "decode"))
    hyp_path = os.path.join(decode_dir, f"valid.step{step}.beam1.hyp.en.txt")
    save_lines(hyp_path, hyps)

    bleu_json = os.path.join(decode_dir, f"valid.step{step}.beam1.bleu.json")
    cmd = [
        sys.executable, os.path.join(os.path.dirname(__file__), "eval_bleu.py"),
        "--src", cfg["data"]["valid_src_txt"],
        "--ref", cfg["data"]["valid_tgt_txt"],
        "--hyp", hyp_path,
        "--spm_model", cfg["data"]["spm_model"],
        "--max_len", str(cfg["data"]["max_len"]),
        "--tokenize", cfg["eval"]["sacrebleu_tokenize"],
        "--out_json", bleu_json,
        "--num_examples", str(cfg["eval"]["num_examples"])
    ]
    subprocess.run(cmd, check=True)  # 不吞 stderr

    rep = json.load(open(bleu_json, "r", encoding="utf-8"))

    return {
        "step": int(step),
        "valid_loss_ce": float(valid_loss_ce),
        "valid_ppl": float(valid_ppl),
        "valid_bleu_beam1": float(rep["overall"]["bleu"]),
        "signature": rep["overall"]["signature"],
        "decode_seconds": float(dt),
        "sent_per_sec": float(len(valid_src_ids) / max(dt, 1e-9)),
        "buckets": rep["buckets"],
        "examples": rep["examples"],
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a=ap.parse_args()
    cfg=load_yaml(a.config)

    init_distributed("nccl")
    is_ddp, rank, local_rank, world_size = ddp_env()
    device=torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available(): torch.cuda.set_device(device)

    set_seed(int(cfg["train"]["seed"])+rank, bool(cfg["train"].get("deterministic", False)))

    run_dir=os.path.join(cfg["run"].get("root","runs"), cfg["run"]["name"])
    if is_main_process():
        ensure_dir(run_dir); ensure_dir(os.path.join(run_dir,"logs")); ensure_dir(os.path.join(run_dir,"checkpoints"))
        save_yaml(os.path.join(run_dir,"config_snapshot.yaml"), cfg)
        env={"time":now_ts(),"cmd":" ".join(sys.argv),"rank":rank,"local_rank":local_rank,"world_size":world_size,
             "torch_version":torch.__version__,
             "cuda":torch.version.cuda if torch.cuda.is_available() else None,
             "device_name":torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else None}
        save_json(os.path.join(run_dir,"env.json"), env)
    barrier()

    train_src=load_ids(cfg["data"]["train_src_ids"]); train_tgt=load_ids(cfg["data"]["train_tgt_ids"])
    model=RNNAttnSeq2Seq(cfg["model"]["vocab_size"], cfg["model"]["emb_dim"], cfg["model"]["hidden_size"],
                        cfg["model"]["num_layers"], cfg["model"]["dropout"], cfg["data"]["pad_id"],
                        cfg["model"]["attn_type"], cfg["model"]["attn_dim"]).to(device)
    if cfg["model"].get("emb_init_npy"):
        load_pretrained_embeddings(model.encoder.emb, cfg["model"]["emb_init_npy"])
        load_pretrained_embeddings(model.decoder.emb, cfg["model"]["emb_init_npy"])

    if world_size>1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model=DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    opt=torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["lr"]))
    scaler=torch.cuda.amp.GradScaler(enabled=bool(cfg["train"]["amp"]))
    loss_fn=nn.CrossEntropyLoss(ignore_index=int(cfg["data"]["pad_id"]))

    train_log = JsonlLogger(os.path.join(run_dir,"logs","train_metrics.jsonl")) if is_main_process() else None
    eval_log  = JsonlLogger(os.path.join(run_dir,"logs","eval_metrics.jsonl")) if is_main_process() else None

    max_steps=int(cfg["train"]["max_steps"]); max_tokens=int(cfg["train"]["max_tokens_per_batch"])
    log_every=int(cfg["train"]["log_every"]); eval_every=int(cfg["train"]["eval_every"])
    clip_norm=float(cfg["train"]["clip_norm"])
    autocast_dtype = torch.bfloat16 if cfg["train"].get("autocast_dtype","bf16")=="bf16" else torch.float16

    best_bleu=-1.0
    step=0; epoch=0
    while step<max_steps:
        epoch+=1
        batches=make_token_batches(train_src, train_tgt, max_tokens=max_tokens, shuffle=True, seed=int(cfg["train"]["seed"])+epoch)
        my_batches=shard_batches(batches, rank, world_size)
        for idxs in my_batches:
            if step>=max_steps: break
            t0=time.time()
            tf_ratio=get_tf_ratio(cfg, step)
            model.train()
            opt.zero_grad(set_to_none=True)

            src, src_mask, tgt_in, tgt_out = collate(idxs, train_src, train_tgt, cfg["data"]["pad_id"], cfg["data"]["bos_id"], cfg["data"]["eos_id"], device)
            B,T=tgt_in.shape
            base_model = model.module if hasattr(model,"module") else model
            enc_out, enc_h = base_model.encoder(src, src_mask)
            state=enc_h
            prev=tgt_in[:,0]
            total_loss=0.0
            tok_count=0

            with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=bool(cfg["train"]["amp"])):
                for t in range(T):
                    logits, state, _ = base_model.decoder.forward_step(prev, state, enc_out, src_mask)
                    gold=tgt_out[:,t]
                    total_loss = total_loss + loss_fn(logits, gold)
                    tok_count += int((gold!=cfg["data"]["pad_id"]).sum().item())
                    if t+1<T:
                        use_tf = (torch.rand((B,), device=device) < tf_ratio)
                        pred=torch.argmax(logits,dim=-1)
                        next_gt=tgt_in[:,t+1]
                        prev=torch.where(use_tf, next_gt, pred)

            loss = (total_loss / T)
            scaler.scale(loss).backward()
            if clip_norm>0:
                scaler.unscale_(opt)
                clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(opt); scaler.update()
            dt=time.time()-t0

            if is_main_process() and step%log_every==0:
                rec={"step":int(step),"epoch":int(epoch),"loss":float(loss.item()),"tf_ratio":float(tf_ratio),
                     "tokens":int(tok_count),"seconds":float(dt),"tokens_per_sec":float(tok_count/max(dt,1e-9)),
                     "attn_type":cfg["model"]["attn_type"]}
                train_log.log(rec); print(rec)

            if is_main_process() and step>0 and step%eval_every==0:
                ev=run_eval(cfg, run_dir, base_model, device, step)
                ev["attn_type"]=cfg["model"]["attn_type"]; ev["tf_schedule"]=cfg["train"]["tf_schedule"]
                eval_log.log(ev); print({"eval_bleu":ev["valid_bleu_beam1"],"step":step,"sig":ev["signature"]})
                if ev["valid_bleu_beam1"]>best_bleu:
                    best_bleu=ev["valid_bleu_beam1"]
                    save_ckpt(os.path.join(run_dir,"checkpoints","best.pt"), base_model, opt, scaler, step, best_bleu, cfg)
                save_ckpt(os.path.join(run_dir,"checkpoints","last.pt"), base_model, opt, scaler, step, best_bleu, cfg)

            step += 1

    if is_main_process():
        train_log.close(); eval_log.close()
        print(f"done best_bleu={best_bleu:.4f} run_dir={run_dir}")

if __name__=="__main__": main()
