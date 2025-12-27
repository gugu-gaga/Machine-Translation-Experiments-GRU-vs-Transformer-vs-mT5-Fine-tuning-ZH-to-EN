#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os as _os, sys as _sys
_ROOT = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _ROOT not in _sys.path:
    _sys.path.insert(0, _ROOT)

import argparse, time, json
import torch
from utils.config import load_yaml
from utils.io import load_ids, load_text, save_lines
from models.transformer_nmt import TransformerNMT, TransformerConfig, load_pretrained_embeddings
from main.bleu_utils import bleu_and_signature, bucket_bleu

@torch.no_grad()
def greedy(model, sp, src_ids, pad_id, bos_id, eos_id, max_new_tokens, device, batch_size):
    model.eval()
    hyps=[]
    for st in range(0,len(src_ids),batch_size):
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
def beam_search(model, sp, src_ids, pad_id, bos_id, eos_id, max_new_tokens, device, beam_size, batch_size, len_penalty):
    model.eval()
    hyps=[]
    for st in range(0, len(src_ids), batch_size):
        batch = src_ids[st:st+batch_size]
        src_seqs=[x+[eos_id] for x in batch]
        B=len(src_seqs); S=max(len(x) for x in src_seqs)
        src=torch.full((B,S),pad_id,dtype=torch.long,device=device)
        src_mask=torch.zeros((B,S),dtype=torch.bool,device=device)
        for i,s in enumerate(src_seqs):
            src[i,:len(s)]=torch.tensor(s,dtype=torch.long,device=device)
            src_mask[i,:len(s)]=True
        memory = model.encode(src, src_mask)
        for i in range(B):
            mem_i = memory[i:i+1]
            sm_i = src_mask[i:i+1]
            beams = [([bos_id], 0.0)]
            for _ in range(max_new_tokens):
                cand=[]
                for toks, logp in beams:
                    if toks[-1]==eos_id:
                        cand.append((toks, logp)); continue
                    tgt = torch.tensor(toks, dtype=torch.long, device=device)[None, :]
                    logits = model.decode(tgt, mem_i, sm_i)[:, -1, :].squeeze(0)
                    lp = torch.log_softmax(logits, dim=-1)
                    top = torch.topk(lp, beam_size)
                    for j in range(beam_size):
                        tid=int(top.indices[j].item()); add=float(top.values[j].item())
                        cand.append((toks+[tid], logp+add))
                def score_fn(item):
                    toks, logp = item
                    L = max(1, len(toks))
                    return logp / (L ** len_penalty if len_penalty != 1.0 else 1.0)
                cand.sort(key=score_fn, reverse=True)
                beams = cand[:beam_size]
                if all(t[-1]==eos_id for t,_ in beams):
                    break
            best = max(beams, key=lambda x: x[1] / (max(1,len(x[0])) ** len_penalty if len_penalty != 1.0 else 1.0))
            out=[]
            for tid in best[0][1:]:
                if tid==eos_id: break
                out.append(tid)
            hyps.append(sp.decode(out))
    return hyps

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", choices=["valid","test"], required=True)
    ap.add_argument("--beam", type=int, default=1)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    a=ap.parse_args()

    cfg=load_yaml(a.config)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    ckpt=torch.load(a.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=cfg["data"]["spm_model"])

    if a.split=="valid":
        src_ids=load_ids(cfg["data"]["valid_src_ids"]); src_txt=load_text(cfg["data"]["valid_src_txt"]); ref=load_text(cfg["data"]["valid_tgt_txt"])
    else:
        src_ids=load_ids(cfg["data"]["test_src_ids"]); src_txt=load_text(cfg["data"]["test_src_txt"]); ref=load_text(cfg["data"]["test_tgt_txt"])

    t0=time.time()
    if a.beam==1:
        hyps=greedy(model, sp, src_ids, int(cfg["data"]["pad_id"]), int(cfg["data"]["bos_id"]), int(cfg["data"]["eos_id"]), a.max_new_tokens, device, batch_size=int(cfg["decode"]["eval_batch_size"]))
    else:
        hyps=beam_search(model, sp, src_ids, int(cfg["data"]["pad_id"]), int(cfg["data"]["bos_id"]), int(cfg["data"]["eos_id"]), a.max_new_tokens, device,
                         beam_size=a.beam, batch_size=int(cfg["decode"]["beam_batch_size"]), len_penalty=float(cfg["decode"].get("len_penalty",1.0)))
    dt=time.time()-t0

    out_dir=_os.path.join(_os.path.dirname(a.config),"decode")
    _os.makedirs(out_dir, exist_ok=True)
    hyp_path=_os.path.join(out_dir,f"{a.split}.beam{a.beam}.hyp.en.txt")
    save_lines(hyp_path, hyps)

    bleu, sig = bleu_and_signature(hyps, ref, tokenize=str(cfg["eval"]["sacrebleu_tokenize"]))
    buckets,_ = bucket_bleu(src_txt, hyps, ref, spm_model=cfg["data"]["spm_model"], max_len=int(cfg["data"]["max_len"]), tokenize=str(cfg["eval"]["sacrebleu_tokenize"]))

    out_json=_os.path.join(out_dir,f"{a.split}.beam{a.beam}.bleu.json")
    with open(out_json,"w",encoding="utf-8") as f:
        json.dump({"overall":{"bleu":bleu,"signature":sig,"n":len(ref)},"buckets":buckets}, f, ensure_ascii=False, indent=2)

    print(f"decode_seconds={dt:.3f} sent_per_sec={len(src_ids)/max(dt,1e-9):.2f}")
    print(f"BLEU={bleu:.4f}")
    print(f"Signature: {sig}")
    print(hyp_path)
    print(out_json)

if __name__=="__main__":
    main()
