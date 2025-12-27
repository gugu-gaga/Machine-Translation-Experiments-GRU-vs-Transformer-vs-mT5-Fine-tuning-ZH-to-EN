#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, sys, time, subprocess, json
import torch
from utils.config import load_yaml
from utils.io import load_ids, save_lines
from utils.common import ensure_dir
from models.rnn_nmt import RNNAttnSeq2Seq, load_pretrained_embeddings
from scripts.train_rnn import greedy_decode

@torch.no_grad()
def beam_search_decode(model, sp, src_ids, pad_id, bos_id, eos_id, max_new_tokens, device, beam_size=4, batch_size=16, len_penalty=1.0):
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
        for i in range(B):
            beams=[([bos_id],0.0,enc_h[:,i:i+1,:].contiguous())]
            for _ in range(max_new_tokens):
                cand=[]
                for toks,logp,state in beams:
                    if toks[-1]==eos_id:
                        cand.append((toks,logp,state)); continue
                    prev=torch.tensor([toks[-1]],device=device)
                    logits,new_state,_=model.decoder.forward_step(prev,state,enc_out[i:i+1],mask[i:i+1])
                    lp=torch.log_softmax(logits,dim=-1).squeeze(0)
                    top=torch.topk(lp, beam_size)
                    for j in range(beam_size):
                        tid=int(top.indices[j].item()); add=float(top.values[j].item())
                        cand.append((toks+[tid], logp+add, new_state))
                cand.sort(key=lambda x: x[1]/((len(x[0])**len_penalty) if len_penalty!=1.0 else 1.0), reverse=True)
                beams=cand[:beam_size]
            best=max(beams, key=lambda x: x[1]/((len(x[0])**len_penalty) if len_penalty!=1.0 else 1.0))
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

    model=RNNAttnSeq2Seq(cfg["model"]["vocab_size"], cfg["model"]["emb_dim"], cfg["model"]["hidden_size"],
                        cfg["model"]["num_layers"], cfg["model"]["dropout"], cfg["data"]["pad_id"],
                        cfg["model"]["attn_type"], cfg["model"]["attn_dim"]).to(device)
    if cfg["model"].get("emb_init_npy"):
        load_pretrained_embeddings(model.encoder.emb, cfg["model"]["emb_init_npy"])
        load_pretrained_embeddings(model.decoder.emb, cfg["model"]["emb_init_npy"])
    ckpt=torch.load(a.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    import sentencepiece as spm
    sp=spm.SentencePieceProcessor(model_file=cfg["data"]["spm_model"])

    if a.split=="valid":
        src_ids=load_ids(cfg["data"]["valid_src_ids"]); src_txt=cfg["data"]["valid_src_txt"]; ref_txt=cfg["data"]["valid_tgt_txt"]
    else:
        src_ids=load_ids(cfg["data"]["test_src_ids"]); src_txt=cfg["data"]["test_src_txt"]; ref_txt=cfg["data"]["test_tgt_txt"]

    t0=time.time()
    if a.beam==1:
        hyps=greedy_decode(model, sp, src_ids, cfg["data"]["pad_id"], cfg["data"]["bos_id"], cfg["data"]["eos_id"], a.max_new_tokens, device, cfg["decode"]["eval_batch_size"])
    else:
        hyps=beam_search_decode(model, sp, src_ids, cfg["data"]["pad_id"], cfg["data"]["bos_id"], cfg["data"]["eos_id"], a.max_new_tokens, device,
                                beam_size=a.beam, batch_size=cfg["decode"]["beam_batch_size"], len_penalty=float(cfg["decode"].get("len_penalty",1.0)))
    dt=time.time()-t0

    run_dir=os.path.dirname(a.config)
    out_dir=ensure_dir(os.path.join(run_dir,"decode"))
    hyp_path=os.path.join(out_dir,f"{a.split}.beam{a.beam}.hyp.en.txt")
    save_lines(hyp_path, hyps)

    bleu_json=os.path.join(out_dir,f"{a.split}.beam{a.beam}.bleu.json")
    cmd=[sys.executable, os.path.join(os.path.dirname(__file__),"eval_bleu.py"),
         "--src", src_txt, "--ref", ref_txt, "--hyp", hyp_path,
         "--spm_model", cfg["data"]["spm_model"], "--max_len", str(cfg["data"]["max_len"]),
         "--tokenize", cfg["eval"]["sacrebleu_tokenize"], "--out_json", bleu_json,
         "--num_examples", str(cfg["eval"]["num_examples"])]
    subprocess.run(cmd, check=True)
    print(f"decode_seconds={dt:.3f} sent_per_sec={len(src_ids)/max(dt,1e-9):.2f}")
    print(hyp_path); print(bleu_json)

if __name__=="__main__": main()
