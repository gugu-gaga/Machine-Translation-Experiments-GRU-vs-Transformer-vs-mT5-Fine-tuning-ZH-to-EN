#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os
def sentence_iter(path):
    with open(path,"r",encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: yield line.split()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--out_dir", default="data/spm")
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--ws", type=int, default=5)
    ap.add_argument("--neg", type=int, default=10)
    ap.add_argument("--threads", type=int, default=8)
    a=ap.parse_args()
    from gensim.models import FastText
    os.makedirs(a.out_dir, exist_ok=True)
    sentences=list(sentence_iter(a.corpus))
    model=FastText(vector_size=a.dim, window=a.ws, min_count=1, sg=1, negative=a.neg,
                  workers=a.threads, min_n=3, max_n=6)
    model.build_vocab(sentences)
    model.train(sentences, total_examples=len(sentences), epochs=a.epochs)
    out=os.path.join(a.out_dir,f"fasttext_sg_dim{a.dim}.model")
    model.save(out)
    print(f"Saved FastText model: {out}")
    print(f"FastText vocab size: {len(model.wv)}")
if __name__=="__main__": main()
