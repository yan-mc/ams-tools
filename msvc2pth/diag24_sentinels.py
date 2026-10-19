#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""task-24 诊断1：定位法国赌神 40k 尾部哨兵值（缓冲区内位置/比特型/物理来源）。"""
import json, os, sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import msvc2pth as M

path = r"F:\ai配音\心月-变声器\心月变声\_internal\modules\法国赌神\法国赌神.msvc"
records, trailer = M.read_records(path)
meta = M.parse_metadata(records[0]); rev = records[2]
runs = M.find_json_runs(rev)
runs2, order, trims, cand, table, json_len = M.search_stitch(rev, int(meta["tensor_count"]))
max_end = table["_max_end"]; tensors = table["tensors"]; tl = {t["key"]: t for t in tensors}
print("max_end", max_end, "runs", runs, "json_len", json_len)
# 当前缓存范围
cache = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_cache.json"), encoding="utf-8"))
entry = None
for k, v in cache.items():
    if "法国赌神" in (v.get("source") or ""):
        entry = v; key = k; break
print("cache key", key[:40], "ranges", entry["shard_ranges"], "perm", entry["perm"], "lens", entry["shard_lengths"])
ranges = entry["shard_ranges"]; perm = entry["perm"]
shards = [rev[a:b] for a, b in ranges]
buf = b"".join(shards[i] for i in perm)
arr = np.frombuffer(buf, dtype="<f2")
print("buf len", len(buf), "=", max_end)
# 非有限 / |v|>20 位置
bad = (~np.isfinite(arr)) | (np.abs(np.where(np.isfinite(arr), arr, 0)) > 20)
idx = np.where(bad)[0]
print("bad values", len(idx))
print("bit patterns:", sorted(set(("%04x" % x) for x in arr[idx].view(np.uint16).tolist())))
# 映射到张量
for t in tensors:
    off = int(t["offset"])//2; ln = int(t["length"])//2
    n = int(bad[off:off+ln].sum())
    if n:
        pos = idx[(idx >= off) & (idx < off+ln)]
        print("  tensor", t["key"], "offset", t["offset"], "len", t["length"], "bad", n, "data_value_pos", (pos*2 - int(t["offset"]))[:6], "...", (pos*2 - int(t["offset"]))[-3:])
# 把 bad 的 buffer 字节位置映射回物理（按 perm 累计）
lens = [len(s) for s in shards]
cum = [0]
for i in perm:
    cum.append(cum[-1] + lens[i])
print("perm", perm, "lens", lens, "cum", cum)
bad_bytes = [p*2 for p in idx]
mapping = []
for p in bad_bytes:
    for j in range(len(perm)):
        if cum[j] <= p < cum[j+1]:
            si = perm[j]
            a, b = ranges[si]
            mapping.append((p, si, a + (p - cum[j])))
            break
print("first mappings (buffer_pos, shard_idx, rev_pos):", mapping[:4])
print("rev_pos min/max:", min(m[2] for m in mapping), max(m[2] for m in mapping))
print("rev bytes sample:", [rev[m[2]:m[2]+2].hex() for m in mapping[:6]])
# 检查 bad 是否连续段
segs = []
start = prev = bad_bytes[0]
for p in bad_bytes[1:]:
    if p == prev + 2:
        prev = p; continue
    segs.append((start, prev)); start = prev = p
segs.append((start, prev))
print("bad byte segments (buffer offsets):", [(a, b, b-a+2) for a, b in segs][:10], "count", len(segs))
