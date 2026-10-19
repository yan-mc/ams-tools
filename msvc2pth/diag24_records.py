#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断9：容器记录结构 + 边界标记搜索。"""
import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import msvc2pth as M
for name, path in [
    ("爱莉希雅", r"F:\ai配音\心月-变声器\心月变声\_internal\modules\爱莉希雅\爱莉希雅.msvc"),
    ("法国赌神", r"F:\ai配音\心月-变声器\心月变声\_internal\modules\法国赌神\法国赌神.msvc"),
]:
    print("="*30, name)
    records, trailer = M.read_records(path)
    print("record lens:", [len(r) for r in records], "trailer", trailer.hex())
    rec0 = records[0]
    print("rec0 head:", repr(rec0[:120]))
    print("rec0 tail:", repr(rec0[-120:]))
    rec1 = records[1]
    pr = sum(1 for x in rec1 if x in (9,10,13) or 32 <= x < 127)/max(1,len(rec1))
    print("rec1 len", len(rec1), "printable", round(pr,3), "head", repr(rec1[:60]), "tail", repr(rec1[-60:]))
    # 搜索 PRV1 类标记（在 rec 里的所有出现）
    for pat in [b"PRV1", b"XINYU", b"MSVC"]:
        for ri, r in enumerate(records):
            pos=[]; i=0
            while True:
                j = r.find(pat, i)
                if j < 0: break
                pos.append(j); i=j+1
            if pos:
                print("  %s in rec%d at %r ctx=%r" % (pat.decode(), ri, pos[:5], r[max(0,pos[0]-8):pos[0]+12]))
    rev = records[2]
    # 所有可打印 run（含 <80）
    allruns=[]; cur=None
    for j,b in enumerate(rev):
        ok = (b in (9,10,13)) or (32 <= b < 127)
        if ok and cur is None: cur=j
        elif not ok and cur is not None:
            if j-cur >= 6: allruns.append((cur,j,rev[cur:j]))
            cur=None
    if cur is not None and len(rev)-cur >= 6: allruns.append((cur,len(rev),rev[cur:]))
    big = [r for r in allruns if r[1]-r[0] >= 80]
    small = [r for r in allruns if r[1]-r[0] < 80]
    print("  rev printable runs >=6:", len(allruns), "big:", len(big), "small count:", len(small))
    print("  small runs sample:", [(a,b,repr(t[:24])) for a,b,t in small[:12]])
