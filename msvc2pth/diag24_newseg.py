#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断6：新分段（g1 起点前移 1 字节，总长恰好==max_end）下 120 排列评估。"""
import itertools, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import msvc2pth as M
path = r"F:\ai配音\心月-变声器\心月变声\_internal\modules\法国赌神\法国赌神.msvc"
refp = r"F:\ai歌曲\RVC20260311\assets\weights\manbo.pth"
records, trailer = M.read_records(path)
meta = M.parse_metadata(records[0]); rev = records[2]
runs = M.find_json_runs(rev)
runs2, order, trims, cand, table, json_len = M.search_stitch(rev, int(meta["tensor_count"]))
max_end = table["_max_end"]; tensors = table["tensors"]; tl = {t["key"]: t for t in tensors}
rk = torch.load(refp, map_location="cpu"); rs = rk["weight"]
refs = {}
for k in rs:
    a = np.nan_to_num(rs[k].float().numpy().astype(np.float64), nan=0, posinf=0, neginf=0)
    s = float(np.std(a))
    if s > 0: refs[k] = float(np.log(s))
# 新分段
ranges = [[500,3008],[4000,4995],[50876,55120565],[55121568,55122568],[55125068,55125568]]
lens = [b-a for a,b in ranges]
print("ranges lens", lens, "sum", sum(lens), "max_end", max_end)
shards = [rev[a:b] for a,b in ranges]
bad_positions = np.zeros(max_end//2, dtype=bool)
rows = []
for perm in itertools.permutations(range(5)):
    buf = b"".join(shards[i] for i in perm)
    arr = np.frombuffer(buf, dtype="<f2")
    fin = np.isfinite(arr)
    bad = (~fin) | (np.abs(np.where(fin, arr, 0)) > 20)
    nbad = int(bad.sum())
    xs=[]; ys=[]; worst=[]
    for t in tensors:
        off=int(t["offset"])//2; ln=int(t["length"])//2
        v = arr[off:off+ln].astype(np.float32)
        if v.size == 0: continue
        s = float(np.nanstd(v))
        rsd = refs.get(t["key"])
        if rsd is not None and np.isfinite(s) and s > 0:
            xs.append(np.log(s)); ys.append(rsd)
        nb = int(bad[off:off+ln].sum())
        if nb: worst.append((t["key"], nb))
    a=np.array(xs); b=np.array(ys); m=np.isfinite(b)
    corr = float(np.corrcoef(a[m],b[m])[0,1])
    rows.append((nbad, -corr, perm, corr, worst[:3]))
rows.sort()
for r in rows[:10]:
    print("  bad=%4d corr=%.4f perm=%r worst=%r" % (r[0], r[3], r[2], r[4]))
