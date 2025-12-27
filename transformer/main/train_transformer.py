#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse, time, math, json
import torch
import torch.nn.functional as F
import torch.nn as nn

from utils.common import set_seed, now_ts, ensure_dir, JsonlLogger, save_json
from utils.config import load_yaml, save_yaml
from utils.io import load_ids, load_text, save_lines
from utils.dist import init_distributed, ddp_env, is_main_process, barrier
from utils.batching import make_token_batches, shard_batches
from models.transformer_nmt import TransformerNMT, TransformerConfig, load_pretrained_embeddings
from main.bleu_utils import bleu_and_signature, bucket_bleu, bucket_id

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

def lr_cosine(step: int, peak_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (peak_lr - min_lr) * cosine

@torch.no_grad()
def greedy_decode(model: TransformerNMT, sp, src_ids, pad_id, bos_id, eos_id, max_new_tokens: int, device, batch_size: int):
    model.eval()
    hyps=[]
    for st in range(0, len(src_ids), batch_size):
        batch=src_ids[st:st+batch_size]
        src_seqs=[x+[eos_id] for x in batch]
        B=len(src_seqs); S=max(len(x) for x in src_seqs)
        src=torch.full((B,S),pad_id,dtype=torch.long,device=device)
        src_mask=torch.zeros((B,S),dtype=torch.bool,device=device)
        for i,s in enumerate(src_seqs):
            src[i,:len(s)]=torch.tensor(s,dtype=torch.long,device=device)
            src_mask[i,:len(s)]=True
        memory = model.encode(src, src_mask)
        out_ids=[[] for _ in range(B)]
        finished=torch.zeros((B,),dtype=torch.bool,device=device)
        tgt = torch.full((B,1), bos_id, dtype=torch.long, device=device)
        for _ in range(max_new_tokens):
            logits = model.decode(tgt, memory, src_mask)
            nxt = torch.argmax(logits[:, -1, :], dim=-1)
            tgt = torch.cat([tgt, nxt[:,None]], dim=1)
            for i in range(B):
                if finished[i]: continue
                tid=int(nxt[i].item())
                if tid==eos_id: finished[i]=True
                else: out_ids[i].append(tid)
            if finished.all(): break
        for i in range(B):
            hyps.append(sp.decode(out_ids[i]))
    return hyps

@torch.no_grad()
def valid_tf_loss(model: TransformerNMT, valid_src_ids, valid_tgt_ids, pad_id, bos_id, eos_id, device, batch_size: int):
    model.eval()
    loss_sum = nn.CrossEntropyLoss(ignore_index=pad_id, reduction="sum")
    total_nll=0.0
    total_tok=0
    for st in range(0, len(valid_src_ids), batch_size):
        idxs=list(range(st, min(st+batch_size, len(valid_src_ids))))
        src, src_mask, tgt_in, tgt_out = collate(idxs, valid_src_ids, valid_tgt_ids, pad_id, bos_id, eos_id, device)
        memory = model.encode(src, src_mask)
        logits = model.decode(tgt_in, memory, src_mask)  # [B,T,V]
        B,T,V = logits.shape
        total_nll += float(loss_sum(logits.view(B*T, V), tgt_out.view(B*T)).item())
        total_tok += int((tgt_out != pad_id).sum().item())
    ce = total_nll / max(1, total_tok)
    ppl = float(torch.exp(torch.tensor(ce)).item())
    return float(ce), float(ppl)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a=ap.parse_args()
    cfg=load_yaml(a.config)

    init_distributed("nccl")
    is_ddp, rank, local_rank, world_size = ddp_env()
    device=torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    set_seed(int(cfg["train"]["seed"])+rank, bool(cfg["train"].get("deterministic", False)))

    run_dir=_os.path.join(cfg["run"].get("root","runs"), cfg["run"]["name"])
    if is_main_process():
        ensure_dir(run_dir); ensure_dir(_os.path.join(run_dir,"logs")); ensure_dir(_os.path.join(run_dir,"checkpoints")); ensure_dir(_os.path.join(run_dir,"decode"))
        save_yaml(_os.path.join(run_dir,"config_snapshot.yaml"), cfg)
        env={"time":now_ts(),"cmd":" ".join(_sys.argv),"rank":rank,"local_rank":local_rank,"world_size":world_size,
             "torch_version":torch.__version__,
             "cuda":torch.version.cuda if torch.cuda.is_available() else None,
             "device_name":torch.cuda.get_device_name(local_rank) if torch.cuda.is_available() else None}
        save_json(_os.path.join(run_dir,"env.json"), env)
    barrier()

    train_src=load_ids(cfg["data"]["train_src_ids"]); train_tgt=load_ids(cfg["data"]["train_tgt_ids"])
    valid_src_ids=load_ids(cfg["data"]["valid_src_ids"]); valid_tgt_ids=load_ids(cfg["data"]["valid_tgt_ids"])
    valid_src_txt=load_text(cfg["data"]["valid_src_txt"]); valid_ref=load_text(cfg["data"]["valid_tgt_txt"])

    mcfg = TransformerConfig(
        vocab_size=int(cfg["model"]["vocab_size"]),
        d_model=int(cfg["model"]["d_model"]),
        n_heads=int(cfg["model"]["n_heads"]),
        d_ff=int(cfg["model"]["d_ff"]),
        n_layers=int(cfg["model"]["n_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        norm_type=str(cfg["model"]["norm_type"]),
        pos_type=str(cfg["model"]["pos_type"]),
        max_len=int(cfg["data"]["max_len"]),
    )
    model = TransformerNMT(mcfg, pad_id=int(cfg["data"]["pad_id"])).to(device)
    emb_init = cfg["model"].get("emb_init_npy", None)
    if emb_init:
        load_pretrained_embeddings(model.emb, emb_init)

    if world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    base_model = model.module if hasattr(model, "module") else model

    opt = torch.optim.AdamW(base_model.parameters(),
                            lr=float(cfg["train"]["peak_lr"]),
                            betas=tuple(cfg["train"]["betas"]),
                            eps=float(cfg["train"]["eps"]),
                            weight_decay=float(cfg["train"]["weight_decay"]))

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"]["amp"]))
    autocast_dtype = torch.bfloat16 if cfg["train"].get("autocast_dtype","bf16")=="bf16" else torch.float16

    pad_id=int(cfg["data"]["pad_id"]); bos_id=int(cfg["data"]["bos_id"]); eos_id=int(cfg["data"]["eos_id"])
    label_smoothing=float(cfg["train"].get("label_smoothing", 0.0))

    def xent_loss(logits, targets):
        if label_smoothing <= 0.0:
            return F.cross_entropy(logits, targets, ignore_index=pad_id, reduction="mean")
        V = logits.size(-1)
        logp = F.log_softmax(logits, dim=-1)
        nll = F.nll_loss(logp, targets, ignore_index=pad_id, reduction="mean")
        smooth = -logp.mean(dim=-1)
        mask = (targets != pad_id).float()
        smooth = (smooth * mask).sum() / mask.sum().clamp_min(1.0)
        return (1.0 - label_smoothing) * nll + label_smoothing * smooth

    train_log = JsonlLogger(_os.path.join(run_dir,"logs","train_metrics.jsonl")) if is_main_process() else None
    eval_log  = JsonlLogger(_os.path.join(run_dir,"logs","eval_metrics.jsonl"))  if is_main_process() else None

    max_steps=int(cfg["train"]["max_steps"])
    log_every=int(cfg["train"]["log_every"])
    eval_every=int(cfg["train"]["eval_every"])
    warmup_steps=int(cfg["train"]["warmup_steps"])
    peak_lr=float(cfg["train"]["peak_lr"])
    min_lr=float(cfg["train"]["min_lr"])
    clip_norm=float(cfg["train"]["clip_norm"])
    tokens_per_gpu=int(cfg["train"]["tokens_per_gpu"])
    grad_accum=int(cfg["train"]["grad_accum"])
    eff_tokens = tokens_per_gpu * grad_accum * world_size

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=cfg["data"]["spm_model"])

    step=0
    epoch=0
    best_bleu=-1.0

    while step < max_steps:
        epoch += 1
        batches = make_token_batches(train_src, train_tgt, max_tokens=tokens_per_gpu, shuffle=True, seed=int(cfg["train"]["seed"])+epoch)
        my_batches = shard_batches(batches, rank, world_size)
        for idxs in my_batches:
            if step >= max_steps: break
            t0=time.time()
            lr = lr_cosine(step, peak_lr, min_lr, warmup_steps, max_steps)
            for pg in opt.param_groups:
                pg["lr"] = lr

            base_model.train()
            opt.zero_grad(set_to_none=True)

            tok_count=0
            total_loss=0.0
            for _ in range(grad_accum):
                src, src_mask, tgt_in, tgt_out = collate(idxs, train_src, train_tgt, pad_id, bos_id, eos_id, device)
                tok_count += int((tgt_out != pad_id).sum().item())
                with torch.autocast(device_type="cuda", dtype=autocast_dtype, enabled=bool(cfg["train"]["amp"])):
                    memory = base_model.encode(src, src_mask)
                    logits = base_model.decode(tgt_in, memory, src_mask)
                    B,T,V = logits.shape
                    loss = xent_loss(logits.view(B*T, V), tgt_out.view(B*T)) / grad_accum
                scaler.scale(loss).backward()
                total_loss += float(loss.item())

            if clip_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(base_model.parameters(), clip_norm)
            scaler.step(opt)
            scaler.update()

            dt=time.time()-t0
            if is_main_process() and (step % log_every == 0):
                rec={"step":int(step),"epoch":int(epoch),"loss":float(total_loss),"lr":float(lr),
                     "tokens":int(tok_count),"seconds":float(dt),"tokens_per_sec":float(tok_count/max(dt,1e-9)),
                     "eff_tokens_per_update":int(eff_tokens),
                     "pos_type":cfg["model"]["pos_type"],"norm_type":cfg["model"]["norm_type"],
                     "d_model":int(cfg["model"]["d_model"]),"n_layers":int(cfg["model"]["n_layers"]),
                     "grad_accum":int(grad_accum),"world_size":int(world_size)}
                train_log.log(rec); print(rec)

            if is_main_process() and (step > 0) and (step % eval_every == 0):
                v_ce, v_ppl = valid_tf_loss(base_model, valid_src_ids, valid_tgt_ids, pad_id, bos_id, eos_id, device, batch_size=int(cfg["decode"]["eval_batch_size"]))
                t1=time.time()
                hyps = greedy_decode(base_model, sp, valid_src_ids, pad_id, bos_id, eos_id, max_new_tokens=int(cfg["decode"]["max_new_tokens_eval"]), device=device, batch_size=int(cfg["decode"]["eval_batch_size"]))
                dsec=time.time()-t1
                bleu, sig = bleu_and_signature(hyps, valid_ref, tokenize=str(cfg["eval"]["sacrebleu_tokenize"]))
                buckets, src_lens = bucket_bleu(valid_src_txt, hyps, valid_ref, spm_model=cfg["data"]["spm_model"], max_len=int(cfg["data"]["max_len"]), tokenize=str(cfg["eval"]["sacrebleu_tokenize"]))
                hyp_path=_os.path.join(run_dir,"decode",f"valid.step{step}.beam1.hyp.en.txt")
                save_lines(hyp_path, hyps)
                ex=[]
                for i in range(min(int(cfg["eval"]["num_examples"]), len(valid_src_txt))):
                    ex.append({"src_zh":valid_src_txt[i],"ref_en":valid_ref[i],"hyp_en":hyps[i],
                               "src_len_subword":int(src_lens[i]),"bucket":bucket_id(int(src_lens[i]), int(cfg["data"]["max_len"]))})
                ev={"step":int(step),"valid_loss_ce":float(v_ce),"valid_ppl":float(v_ppl),
                    "valid_bleu_beam1":float(bleu),"signature":sig,
                    "decode_seconds":float(dsec),"sent_per_sec":float(len(valid_src_ids)/max(dsec,1e-9)),
                    "buckets":buckets,"examples":ex}
                eval_log.log(ev); print({"eval_bleu":bleu,"step":step,"sig":sig,"valid_ppl":v_ppl})
                ckpt={"model":base_model.state_dict(),"step":int(step),"best_bleu":float(max(best_bleu,bleu)),"cfg":cfg}
                torch.save(ckpt, _os.path.join(run_dir,"checkpoints","last.pt"))
                if bleu > best_bleu:
                    best_bleu = bleu
                    torch.save(ckpt, _os.path.join(run_dir,"checkpoints","best.pt"))

            step += 1

    if is_main_process():
        train_log.close(); eval_log.close()
        print(f"done best_bleu={best_bleu:.4f} run_dir={run_dir}")

if __name__=="__main__":
    main()
