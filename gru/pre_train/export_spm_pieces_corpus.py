#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os
def load_lines(p):
    with open(p,"r",encoding="utf-8") as f: return [x.rstrip("\n") for x in f]
def write_lines(p, lines):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p,"w",encoding="utf-8") as f:
        for x in lines: f.write(x+"\n")
def encode_lines(sp, lines):
    out=[]
    for s in lines:
        out.append(" ".join(sp.encode(s, out_type=str)))
    return out
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--spm_model", required=True)
    ap.add_argument("--clean_dir", default="data/clean")
    ap.add_argument("--out_dir", default="data/spm_corpus")
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--join_src_tgt", action="store_true")
    a=ap.parse_args()
    import sentencepiece as spm
    sp=spm.SentencePieceProcessor(model_file=a.spm_model)
    for split in a.splits:
        src=load_lines(os.path.join(a.clean_dir,f"{split}.src"))
        tgt=load_lines(os.path.join(a.clean_dir,f"{split}.tgt"))
        src_p=encode_lines(sp, src)
        tgt_p=encode_lines(sp, tgt)
        write_lines(os.path.join(a.out_dir,f"{split}.src.pieces"), src_p)
        write_lines(os.path.join(a.out_dir,f"{split}.tgt.pieces"), tgt_p)
        if a.join_src_tgt:
            write_lines(os.path.join(a.out_dir,f"{split}.joint.pieces"), src_p+tgt_p)
        print(f"[{split}] wrote pieces -> {a.out_dir}")
if __name__=="__main__": main()
