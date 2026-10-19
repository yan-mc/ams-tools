#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断4：每个 gap 的边界微移对 fp16 合理性影响；gap1 是否 1 字节错位。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import msvc2pth as M

path = r"F:\ai配音\心月-变声器\心月变声\_internal\modules\法国赌神\法国赌神.msvc"
records, trailer = M.read_records(path)
meta = M.parse_metadata(records[0]); rev = records[2]
runs = M.find_json_runs(rev)
runs2, order, trims, cand, table, json_len = M.search_stitch(rev, int(meta["tensor_count"]))
max_end = table["_max_end"]; tensors = table["tensors"]
print("sum(lengths)=", sum(int(t["length"]) for t in tensors), "max_end=", max_end)
# 检查 offset 是否连续
offs = [int(t["offset"]) for t in tensors]
ends = [int(t["offset"])+int(t["length"]) for t in tensors]
gapsz = [offs[i+1] - ends[i] for i in range(len(offs)-1)]
print("tensor offset gaps nonzero:", [(i, g) for i, g in enumerate(gapsz) if g != 0][:5], "count", sum(1 for g in gapsz if g))
gaps = []
for i in range(len(runs)-1): gaps.append((runs[i][1], runs[i+1][0]))
gaps.append((runs[-1][1], len(rev)))

def stats(seg):
    n = (len(seg)//2)*2
    arr = np.frombuffer(seg[:n], dtype="<f2").astype(np.float32)
    fin = np.isfinite(arr)
    small = np.abs(np.where(fin, arr, 1e9)) < 1.0
    return fin.mean(), small.mean(), float(np.nanmax(np.abs(arr))) if len(arr) else 0.0

for gi in (0,1,3,4):
    a,b = gaps[gi]
    print("gap%d base:" % gi, "len", b-a)
    for s in range(-4,5):
        for e in range(-2,3):
            seg = rev[a+s:b+e]
            if len(seg) <= 16: continue
            fin, small, mx = stats(seg)
            if fin >= 0.999 and small >= 0.99:
                print("  shift start=%+d end=%+d len=%d finite=%.4f small=%.4f max=%.3g" % (s, e, len(seg), fin, small, mx))
