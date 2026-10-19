#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""写入已校准的分片范围缓存（M0.5 结果；每文件 sha256 缓存）。"""
import hashlib, json, os, sys

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()

CASES = [
    ("爱莉希雅", r"F:\ai配音\心月-变声器\心月变声\_internal\modules\爱莉希雅\爱莉希雅.msvc",
     r"F:\ai歌曲\RVC20260311\assets\weights\爱莉希雅.pth",
     [[500, 3008], [4002, 4999], [50832, 57479819], [57480821, 57481820], [57484321, 57484818]],
     [2, 4, 0, 1, 3], 0.6501, 0.6500, 57433988, 57433989, -1, "layout_prior_48k"),
    ("法国赌神", r"F:\ai配音\心月-变声器\心月变声\_internal\modules\法国赌神\法国赌神.msvc",
     r"F:\ai歌曲\RVC20260311\assets\weights\manbo.pth",
     [[500, 3008], [4001, 4995], [50878, 55120566], [55121568, 55122568], [55125066, 55125568]],
     [2, 4, 0, 1, 3], 1.0, 0.9995, 55074692, 55074691, 1, "even_edge_big_start_pad2"),
    ("曼波Pro", r"F:\ai配音\心月-变声器\心月变声\_internal\modules\曼波Pro\苏苏.msvc",
     r"F:\ai歌曲\RVC20260311\assets\weights\爱莉希雅.pth",
     [[500, 3002], [4000, 5000], [50887, 57479873], [57480875, 57481875], [57484375, 57484875]],
     [2, 4, 0, 1, 3], 0.6571, 0.6571, 57433988, 57433995, -7, "edge_shrink_shard0"),
]
cache = {}
for name, src, ref, ranges, perm, score, corr, max_end, base_total, diff, note in CASES:
    key = "%s|%s" % (sha(src), sha(ref))
    lens = [b - a for a, b in ranges]
    assert sum(lens) == max_end, (name, lens, max_end)
    cache[key] = {
        "big_start_delta": None,
        "tail_trim": None,
        "shard_ranges": ranges,
        "perm": perm,
        "score": score,
        "finite_ratio": 1.0,
        "ref_corr": corr,
        "shard_lengths": lens,
        "max_end": max_end,
        "base_total": base_total,
        "diff": diff,
        "calibration_note": note,
        "tool_version": "0.3.0",
        "source_sha256": sha(src),
        "source": src,
        "reference_pth": ref,
    }
    print(name, lens, "sum", sum(lens), "corr", corr)
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_cache.json")
open(p, "w", encoding="utf-8").write(json.dumps(cache, ensure_ascii=False, indent=2))
print("cache ->", p, "entries", len(cache))
