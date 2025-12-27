#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os
import numpy as np

def l2_normalize_rows(mat, eps=1e-12):
    # Safe, no broadcasting pitfalls: (V,512)/(V,1) -> (V,512)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, eps)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--spm_model", required=True)
    ap.add_argument("--ft_model", required=True)
    ap.add_argument("--out_npy", required=True)
    ap.add_argument("--out_meta", required=True)
    ap.add_argument("--pad_zero", action="store_true")
    ap.add_argument("--special_mean", action="store_true")
    ap.add_argument("--l2_normalize", action="store_true")
    a=ap.parse_args()

    import sentencepiece as spm
    from gensim.models import FastText
    sp=spm.SentencePieceProcessor(model_file=a.spm_model)
    ft=FastText.load(a.ft_model)

    V=sp.get_piece_size()
    dim=ft.wv.vector_size
    mat=np.zeros((V,dim), dtype=np.float32)
    mean_vec=ft.wv.vectors.mean(axis=0).astype(np.float32)

    pad_id=sp.pad_id(); unk_id=sp.unk_id(); bos_id=sp.bos_id(); eos_id=sp.eos_id()

    for i in range(V):
        piece=sp.id_to_piece(i)
        if a.pad_zero and i==pad_id:
            mat[i]=0.0
            continue
        if piece in ft.wv:
            mat[i]=ft.wv[piece].astype(np.float32)
        else:
            mat[i]=mean_vec

    if a.special_mean:
        for sid in (unk_id, bos_id, eos_id):
            mat[sid]=mean_vec
        if a.pad_zero:
            mat[pad_id]=0.0

    if a.l2_normalize:
        mat=l2_normalize_rows(mat)

    os.makedirs(os.path.dirname(a.out_npy), exist_ok=True)
    np.save(a.out_npy, mat)

    meta={
        "spm_model": a.spm_model,
        "ft_model": a.ft_model,
        "shape": [int(V), int(dim)],
        "pad_id": int(pad_id),
        "unk_id": int(unk_id),
        "bos_id": int(bos_id),
        "eos_id": int(eos_id),
        "pad_zero": bool(a.pad_zero),
        "special_mean": bool(a.special_mean),
        "l2_normalize": bool(a.l2_normalize),
    }
    with open(a.out_meta,"w",encoding="utf-8") as f:
        json.dump(meta,f,ensure_ascii=False,indent=2)

    print(f"Saved: {a.out_npy}  shape={mat.shape}")
    print(f"Saved: {a.out_meta}")

if __name__=="__main__": main()
