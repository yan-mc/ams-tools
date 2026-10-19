#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断8：修正映射（字节位置）定位 bad 值来源分段。"""
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
cum = [0]
for i in perm: cum.append(cum[-1]+lens[i])
print("perm", perm, "lens", lens, "cum", cum, "bad", len(idx), "buffer byte range", int(idx.min())*2, int(idx.max())*2)
segmap = {}
for p in idx:
    bp = int(p)*2
    for j in range(5):
        if cum[j] <= bp < cum[j+1]:
            si = perm[j]; a,b = ranges[si]
            segmap.setdefault(si, []).append(a + (bp - cum[j]))
            break
for si, ps in sorted(segmap.items()):
    ps = np.array(ps)
    print("  segment", si, "range", ranges[si], "len", lens[si], "count", len(ps), "rev_pos", ps.min(), "..", ps.max())
    p0 = int(ps[0]); print("    ctx:", repr(rev[p0-12:p0+28]))
    # 该段内 bad 的密度
# 打印所有 segment 2 (big gap) 内 bad? none per above
# 反查：bad 值对应的张量
for t in tensors:
    off=int(t["offset"])//2; ln=int(t["length"])//2
    nb = int(bad[off:off+ln].sum())
    if nb:
        print("  tensor", t["key"], "bad", nb, "offset", t["offset"], "len", t["length"])
