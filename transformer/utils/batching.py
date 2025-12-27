from typing import List
import random

def make_token_batches(src_ids: List[List[int]], tgt_ids: List[List[int]], max_tokens: int, shuffle: bool, seed: int) -> List[List[int]]:
    assert len(src_ids) == len(tgt_ids)
    n = len(src_ids)
    lens = [(i, max(len(src_ids[i]), len(tgt_ids[i]))) for i in range(n)]
    lens.sort(key=lambda x: x[1])
    rng = random.Random(seed)
    if shuffle:
        bucket = 1024
        for b in range(0, n, bucket):
            chunk = lens[b:b+bucket]
            rng.shuffle(chunk)
            lens[b:b+bucket] = chunk
    batches, cur, cur_tok = [], [], 0
    for i, L in lens:
        if cur and cur_tok + L > max_tokens:
            batches.append(cur); cur=[i]; cur_tok=L
        else:
            cur.append(i); cur_tok += L
    if cur: batches.append(cur)
    if shuffle: rng.shuffle(batches)
    return batches

def shard_batches(batches: List[List[int]], rank: int, world_size: int) -> List[List[int]]:
    return batches[rank::world_size]
