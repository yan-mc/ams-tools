#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断5：搜索 rev 中 JSON 关键字位置，判断 gap 内是否藏有 JSON 片段。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import msvc2pth as M
path = r"F:\ai配音\心月-变声器\心月变声\_internal\modules\法国赌神\法国赌神.msvc"
records, trailer = M.read_records(path)
rev = records[2]
runs = M.find_json_runs(rev)
print("runs:", runs)
for pat in [b"PRV1", b"RVC_PRV1", b"runtime_format", b"dtype", b"checksum", b"storage_mode", b"encryption_level", b"model_id", b"timestamp", b"tensor_count", b'"shape"', b'"offset"', b'"length"']:
    pos=[]; i=0
    while True:
        j = rev.find(pat, i)
        if j < 0: break
        pos.append(j); i = j+1
    inbox = [p for p in pos if any(a <= p < b for a,b in runs)]
    out = [p for p in pos if not any(a <= p < b for a,b in runs)]
    print("%-18s total=%d in_runs=%d out=%r" % (pat.decode(), len(pos), len(inbox), out[:10]))
# 打印 gap0 尾部与周围
a,b = runs[0][1], runs[1][0]
print("gap0 tail region rev[%d:%d]:" % (b-40, b), repr(rev[b-40:b]))
# 在 gap0 中所有连续可打印长度>=8 的段
segs=[]; cur=None
for j in range(a,b):
    ok = rev[j] in (9,10,13) or 32 <= rev[j] < 127
    if ok and cur is None: cur=j
    elif not ok and cur is not None:
        if j-cur >= 8: segs.append((cur,j,rev[cur:j]))
        cur=None
if cur is not None: segs.append((cur,b,rev[cur:b]))
print("gap0 printable segs>=8:")
for s in segs: print("  ", s[0], s[1], repr(s[2]))
