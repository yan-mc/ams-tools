#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断3：gap 内容性质 + 几种排列的哨兵/相关/首尾张量统计。"""
import os, sys
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

gaps = []
for i in range(len(runs)-1): gaps.append((runs[i][1], runs[i+1][0]))
gaps.append((runs[-1][1], len(rev)))
print("gap nature check (fp16/反转后文本):")
for gi in (0,1,3,4):
    a,b = gaps[gi]; seg = rev[a:b]
    n = (len(seg)//2)*2
    arr = np.frombuffer(seg[:n], dtype="<f2").astype(np.float32)
    fin = np.isfinite(arr)
    small = np.abs(np.where(fin, arr, 1e9)) < 1.0
    print(" gap%d len=%d fp16: finite=%.3f |v|<1=%.3f max=%.3g" % (gi, len(seg), fin.mean(), small.mean(), np.nanmax(np.abs(arr))))
    rseg = seg[::-1]
    pr = sum(1 for x in rseg if x in (9,10,13) or 32 <= x < 127)/len(rseg)
    try:
        s = rseg.decode("utf-8"); ok = True
    except Exception:
        ok = False
    print("   reversed bytes: printable=%.3f utf8=%s head=%r" % (pr, ok, rseg[:48]))
    print("   reversed+second-order? ", (seg[::-1][:0]))
    # 尝试逐字节反转后取偶数位置? 跳过

# 几种排列
base_ranges = [[500,3008],[4001,4995],[50878,55120566],[55121568,55122568],[55125066,55125568]]
lens = [b-a for a,b in base_ranges]
print("lens", lens, "sum", sum(lens), "max_end", max_end)
shards = [rev[a:b] for a,b in base_ranges]
def evalperm(perm, label):
    buf = b"".join(shards[i] for i in perm)
    arr = np.frombuffer(buf, dtype="<f2")
    bad = (~np.isfinite(arr)) | (np.abs(np.where(np.isfinite(arr), arr, 0)) > 20)
    xs=[]; ys=[]
    for t in tensors:
        off=int(t["offset"])//2; ln=int(t["length"])//2
        v = arr[off:off+ln].astype(np.float32)
        if v.size == 0: continue
        s = float(np.nanstd(v))
        if not np.isfinite(s) or s <= 0: continue
        xs.append(np.log(s)); ys.append(refs.get(t["key"], np.nan))
    a=np.array(xs); b=np.array(ys); m=np.isfinite(b)
    corr = float(np.corrcoef(a[m],b[m])[0,1])
    # 首/尾张量
    first = tensors[0]; last = tensors[-1]
    def st(t):
        off=int(t["offset"])//2; ln=int(t["length"])//2
        v = arr[off:off+ln].astype(np.float32)
        return (float(np.nanstd(v)), int(((~np.isfinite(v)) | (np.abs(np.where(np.isfinite(v),v,0)) > 20)).sum()))
    print("  %-28s bad=%4d corr=%.4f first(%s)=%r last(%s)=%r" % (label, int(bad.sum()), corr, first["key"], st(first), last["key"], st(last)))

for perm in [(0,1,2,3,4),(2,4,0,1,3),(0,1,3,2,4),(0,1,3,4,2),(4,0,1,3,2),(0,1,2,4,3)]:
    evalperm(perm, str(perm))
