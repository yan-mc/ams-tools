#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断7：新分段下 perm(2,1,0,4,3) 的 bad 值精确位置与来源分段。"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import msvc2pth as M
path = r"F:\ai配音\心月-变声器\心月变声\_internal\modules\法国赌神\法国赌神.msvc"
records, trailer = M.read_records(path)
meta = M.parse_metadata(records[0]); rev = records[2]
runs = M.find_json_runs(rev)
runs2, order, trims, cand, table, json_len = M.search_stitch(rev, int(meta["tensor_count"]))
max_end = table["_max_end"]; tensors = table["tensors"]; tl = {t["key"]: t for t in tensors}
ranges = [[500,3008],[4000,4995],[50876,55120565],[55121568,55122568],[55125068,55125568]]
lens = [b-a for a,b in ranges]
perm = (2,1,0,4,3)
shards = [rev[a:b] for a,b in ranges]
buf = b"".join(shards[i] for i in perm)
arr = np.frombuffer(buf, dtype="<f2")
fin = np.isfinite(arr); bad = (~fin) | (np.abs(np.where(fin, arr, 0)) > 20)
idx = np.where(bad)[0]
print("bad", len(idx))
# 映射到分段
cum = [0]
for i in perm: cum.append(cum[-1]+lens[i])
print("perm", perm, "lens", lens, "cum", cum)
segmap = {}
for p in idx:
    for j in range(5):
        if cum[j] <= p < cum[j+1]:
            si = perm[j]
            a,b = ranges[si]
            segmap.setdefault(si, []).append(a + (p - cum[j])*1)
            break
for si, ps in sorted(segmap.items()):
    ps = np.array(ps)
    print("  segment", si, "count", len(ps), "rev_pos min", ps.min(), "max", ps.max(), "first", ps[:5])
# 打印这些 rev 位置的字节上下文
si = sorted(segmap)[0]; ps = np.array(segmap[si])
p0 = int(ps[0]); print("context rev[%d:%d]:" % (p0-16, p0+40), repr(rev[p0-16:p0+40]))
# 检查 g0 内部 bad: g0 自身 fp16 统计
g0 = rev[500:3008]
a0 = np.frombuffer(g0[:2508], dtype="<f2").astype(np.float32)
print("g0 fp16 bad count", int(((~np.isfinite(a0)) | (np.abs(np.where(np.isfinite(a0), a0, 0)) > 20)).sum()), "of", len(a0))
