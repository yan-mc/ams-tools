#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""task-18 诊断：3 个 .msvc 的 JSON 片段/trim/数据分片边界现状。"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import msvc2pth as M

MODELS = [
    ("爱莉希雅", r"F:\ai配音\心月-变声器\心月变声\_internal\modules\爱莉希雅\爱莉希雅.msvc"),
    ("法国赌神", r"F:\ai配音\心月-变声器\心月变声\_internal\modules\法国赌神\法国赌神.msvc"),
    ("曼波Pro", r"F:\ai配音\心月-变声器\心月变声\_internal\modules\曼波Pro\苏苏.msvc"),
]
for name, path in MODELS:
    print("=" * 30, name, os.path.getsize(path))
    records, trailer = M.read_records(path)
    print("records:", [len(r) for r in records], "trailer:", trailer.hex())
    meta = M.parse_metadata(records[0])
    print("meta: tensor_count=%s sr=%s info=%s model_id=%s" % (meta.get("tensor_count"), meta.get("sr"), meta.get("info"), meta.get("model_id")))
    rev = records[2]
    runs = M.find_json_runs(rev)
    print("rev_len:", len(rev), "runs:", [(a, b, b - a) for a, b in runs])
    # 每 run 首尾 12 字节，判断污染
    for i, (a, b) in enumerate(runs):
        print("  run%d [%d,%d): head=%r tail=%r" % (i, a, b, rev[a:a+12], rev[b-12:b]))
    try:
        runs2, order, trims, cand, table, json_len = M.search_stitch(rev, int(meta.get("tensor_count", 457)))
        print("stitch OK order=%r trims=%r json_len=%d max_end=%d" % (order, trims, json_len, table["_max_end"]))
        # 用 trims 还原各片段的真实 span（物理坐标）
        pieces = []
        for k, idx in enumerate(order):
            r = runs[idx]
            s = 0 if k == 0 else trims[2 * (k - 1) + 1]
            e = len(r)
            if k < len(order) - 1:
                e -= trims[2 * k]
            else:
                e -= trims[-1]
            pieces.append((idx, r[0] + s, r[0] + e))
        print("trimmed spans(physical run idx):", pieces)
        # 物理顺序的 5 个数据分片（互补区间）
        spans = sorted(runs)
        trimmed = {}
        for idx, a, b in pieces:
            trimmed[idx] = (a, b)
        shards = []
        cur = 0
        for i, (a, b) in enumerate(spans):
            ta, tb = trimmed[i]
            if ta > cur:
                shards.append((cur, ta))
            cur = tb
        if cur < len(rev):
            shards.append((cur, len(rev)))
        print("physical shards (untrimmed-gap view):", [(a, b, b - a) for a, b in shards])
        total = sum(b - a for a, b in shards)
        print("shard_total=%d max_end=%d excess=%d" % (total, table["_max_end"], total - table["_max_end"]))
    except Exception as e:
        print("stitch FAIL:", repr(e))
