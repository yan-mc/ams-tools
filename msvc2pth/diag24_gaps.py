#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""诊断2：检查法国赌神各 gap 区间内容（是否为 JSON 文本/UTF-8/其他）。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import msvc2pth as M

path = r"F:\ai配音\心月-变声器\心月变声\_internal\modules\法国赌神\法国赌神.msvc"
records, trailer = M.read_records(path)
rev = records[2]
runs = M.find_json_runs(rev)
print("runs:", runs)
gaps = []
for i in range(len(runs)-1):
    gaps.append((runs[i][1], runs[i+1][0]))
gaps.append((runs[-1][1], len(rev)))
for gi, (a, b) in enumerate(gaps):
    seg = rev[a:b]
    printable = sum(1 for x in seg if x in (9,10,13) or 32 <= x < 127)
    print("gap%d [%d,%d) len=%d printable=%.3f head=%r tail=%r" % (gi, a, b, len(seg), printable/len(seg), seg[:60], seg[-40:]))
    # UTF-8 尝试
    if gi in (0, 1):
        try:
            s = seg.decode("utf-8")
            print("   utf8 ok:", repr(s[:200]))
        except Exception as e:
            print("   utf8 fail:", e)
