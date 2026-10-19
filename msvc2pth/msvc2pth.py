#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AMS MSVC -> RVC v2 .pth converter (M0 spike).

仅处理用户本地合法持有的 .msvc 文件；不包含 .cyd 解密，不生成 .index。
格式参考: 06_方案设计/../05_新方案探讨/01_心月格式深拆/格式.md
运行环境: 需要 numpy + torch (可使用 RVC 整合包自带 runtime/python.exe)
"""
import argparse
import hashlib
import itertools
import json
import os
import struct
import sys

import numpy as np

MAGIC = b"MSVC"
TAG = b"XINYU_AI"
TRAILER_LEN = 16
MIN_RUN = 80


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reference(path):
    """加载参考 RVC 小模型（同架构、不同训练模型），用于键集校验与分片顺序判定。"""
    import shutil
    import tempfile
    import torch
    fd, tmp = tempfile.mkstemp(suffix=".pth", prefix="ams_ref_")
    os.close(fd)
    try:
        shutil.copyfile(path, tmp)
        ckpt = torch.load(tmp, map_location="cpu")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return ckpt


def read_records(path):
    data = open(path, "rb").read()
    if data[:4] != MAGIC or data[6:14] != TAG:
        raise ValueError("不是 MSVC/XINYU_AI 容器")
    pos = 14
    records = []
    while pos + 4 <= len(data) - TRAILER_LEN:
        (length,) = struct.unpack_from("<I", data, pos)
        if length <= 0 or pos + 4 + length > len(data):
            raise ValueError("记录长度非法 @%d" % pos)
        payload = data[pos + 4:pos + 4 + length]
        records.append(payload[::-1])
        pos += 4 + length
    if pos != len(data) - TRAILER_LEN:
        raise ValueError("记录区未对齐: pos=%d len=%d" % (pos, len(data)))
    return records, data[-TRAILER_LEN:]


def parse_metadata(seg0):
    text = seg0.decode("utf-8", "ignore")
    dec = json.JSONDecoder()
    i = 0
    while i < len(text):
        j = text.find("{", i)
        if j < 0:
            break
        try:
            obj, end = dec.raw_decode(text[j:])
            if isinstance(obj, dict) and "runtime_format" in obj:
                return obj
            i = j + end
        except Exception:
            i = j + 1
    raise ValueError("未找到 metadata JSON")


def find_json_runs(rev):
    runs = []
    cur = None
    for j, b in enumerate(rev):
        ok = (b in (9, 10, 13)) or (32 <= b < 127)
        if ok and cur is None:
            cur = j
        elif not ok and cur is not None:
            if j - cur >= MIN_RUN:
                runs.append((cur, j))
            cur = None
    if cur is not None and len(rev) - cur >= MIN_RUN:
        runs.append((cur, len(rev)))
    return runs


def validate_table(obj, expected, expected_keys=None):
    ts = obj.get("tensors")
    if not isinstance(ts, list) or len(ts) != expected:
        return False
    keys = []
    total = 0
    prev = -1
    for t in ts:
        if not isinstance(t, dict):
            return False
        if not all(k in t for k in ("key", "dtype", "shape", "offset", "length")):
            return False
        n = 1
        for s in t["shape"]:
            n *= int(s)
        if n * 2 != int(t["length"]):
            return False
        off = int(t["offset"])
        if off <= prev:
            return False
        prev = off
        total = max(total, off + int(t["length"]))
        keys.append(t["key"])
    if len(set(keys)) != len(keys):
        return False
    if expected_keys is not None and set(keys) != set(expected_keys):
        return False
    obj["_max_end"] = total
    return True


def build_candidate(runs, order, trims, last_end_trim):
    pieces = []
    for k, idx in enumerate(order):
        r = runs[idx]
        s = 0
        e = len(r)
        if k > 0:
            s = trims[2 * (k - 1) + 1]
        if k < len(order) - 1:
            e -= trims[2 * k]
        else:
            e -= last_end_trim
        if s >= e:
            return None
        pieces.append(r[s:e])
    return b"".join(pieces)


def search_stitch(rev, expected_tensors, expected_keys=None):
    runs = find_json_runs(rev)
    if len(runs) != 5:
        raise ValueError("期望 5 个 JSON 片段，实际 %d（无法安全解析）" % len(runs))
    runs_b = [rev[a:b] for a, b in runs]
    first = [i for i, r in enumerate(runs_b) if r.startswith(b'{"runtime_format"')]
    last = [i for i, r in enumerate(runs_b) if r.rstrip().endswith(b"}]}")]
    if len(first) != 1 or len(last) != 1 or first[0] == last[0]:
        raise ValueError("无法确定 JSON 首/尾片段")
    fi, li = first[0], last[0]
    middles = [i for i in range(5) if i not in (fi, li)]

    # 范围按“已知格式”收紧；失败再放大。
    ranges = [
        (4, 6),   # boundary0: r1 end 0..3, r3 start 0..5
        (2, 3),   # boundary1
        (2, 3),   # boundary2
        (2, 7),   # boundary3: last start 0..6
    ]
    orders = [[fi] + list(p) + [li] for p in itertools.permutations(middles)]
    # 已知格式的顺序优先（实测顺序：first, 第3片段, 第1片段, 第5片段, last）
    known = [fi, middles[1], middles[0], middles[2], li]
    if known in orders:
        orders.remove(known)
        orders.insert(0, known)

    for order in orders:
        for trims in itertools.product(*[range(n) for pair in ranges for n in pair]):
            for last_trim in (0, 1):
                cand = build_candidate(runs_b, order, trims, last_trim)
                if not cand or not cand.startswith(b"{") or not cand.rstrip().endswith(b"}"):
                    continue
                try:
                    obj = json.loads(cand.decode("utf-8"))
                except Exception:
                    continue
                if not validate_table(obj, expected_tensors, expected_keys):
                    continue
                json_len = len(cand)
                data_len = len(rev) - json_len
                if obj["_max_end"] > data_len:
                    continue
                if data_len - obj["_max_end"] > 64:
                    # 允许少量 padding；超过 64B 视为拼接不可信
                    continue
                return runs, order, list(trims) + [last_trim], cand, obj, json_len
    raise ValueError("无法恢复张量表（拼接搜索失败）")


def build_data_shards(rev, runs, target_len):
    """按原始 JSON 片段边界切出数据分片（物理顺序），并用尾部 padding 校准总长。

    实测心月格式：JSON 片段的“可打印 run”两端会混入少量数据字节；
    中间数据分片按固定规则切分，最后一个分片长度由 target_len 反推，
    从而适配不同模型的尾部 padding（0~N 字节）。
    """
    spans = sorted(runs)
    if len(spans) != 5:
        raise ValueError("数据分片需要 5 个 JSON 片段")
    gaps = []
    if spans[0][0] > 0:
        gaps.append(rev[:spans[0][0]])
    gaps.append(rev[spans[0][1]:spans[1][0]])
    gaps.append(rev[spans[1][1]:spans[2][0]])
    gaps.append(rev[spans[2][1] - 1:spans[3][0]])
    gaps.append(rev[spans[3][1]:spans[4][0]])
    fixed = sum(len(g) for g in gaps)
    need_last = target_len - fixed
    if need_last <= 0:
        raise ValueError("数据分片总长计算失败: fixed=%d target=%d" % (fixed, target_len))
    last_start = spans[4][1]
    available = len(rev) - last_start
    pad = available - need_last
    if pad < 0 or pad > 64:
        raise ValueError("尾部 padding 异常: pad=%d" % pad)
    gaps.append(rev[last_start:len(rev) - pad])
    total = sum(len(g) for g in gaps)
    if total != target_len:
        raise ValueError("数据分片总长不一致: %d vs %d" % (total, target_len))
    return gaps


def order_shards(shards, tensors, max_end, ref_weight=None):
    """确定数据分片的逻辑顺序。

    有参考模型时：按同名张量的 log(std) 相关系数排序（最可靠）。
    无参考模型时：退化为 fp16 合理性评分。
    """
    best = None
    keys = [t["key"] for t in tensors]
    ref_stats = None
    if ref_weight is not None:
        ref_stats = {}
        for k in keys:
            if k in ref_weight:
                rv = ref_weight[k].float().numpy().astype(np.float32)
                sd = float(np.std(rv))
                if sd > 0:
                    ref_stats[k] = float(np.log(sd))
    sample = tensors if len(tensors) <= 96 else tensors[::max(1, len(tensors) // 96)]
    for perm in itertools.permutations(range(len(shards))):
        buf = b"".join(shards[i] for i in perm)
        if len(buf) < max_end:
            continue
        buf = buf[:max_end]
        arr = np.frombuffer(buf, dtype="<f2")
        if ref_stats is not None and len(ref_stats) > 50:
            xs, ys = [], []
            for t in sample:
                k = t["key"]
                if k not in ref_stats:
                    continue
                off = int(t["offset"]) // 2
                ln = int(t["length"]) // 2
                v = arr[off:off + ln].astype(np.float32)
                if v.size == 0 or not np.isfinite(v).all():
                    continue
                sd = float(np.std(v))
                if sd > 0:
                    xs.append(np.log(sd))
                    ys.append(ref_stats[k])
            score = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 10 else -1.0
        else:
            score = 0.0
            for t in sample:
                off = int(t["offset"]) // 2
                ln = int(t["length"]) // 2
                v = arr[off:off + ln]
                if v.size == 0:
                    continue
                finite = float(np.isfinite(v).mean())
                small = float((np.abs(v.astype(np.float32)) < 100).mean())
                score += finite + small
        if best is None or score > best[0]:
            best = (score, perm, buf)
    if best is None:
        raise ValueError("数据分片无法拼接")
    return best[1], best[2], best[0]


def convert(path, out_dir=None, voice_name=None, report_path=None, reference_pth=None):
    import torch  # 延迟导入，便于 --help 在无 torch 环境运行

    ref_weight = None
    expected_keys = None
    if reference_pth:
        ref = load_reference(reference_pth)
        ref_weight = ref.get("weight", ref)
        expected_keys = list(ref_weight.keys())

    records, trailer = read_records(path)
    if len(records) < 3:
        raise ValueError("记录数不足（需要 metadata/unknown/data 三段）")
    metadata = parse_metadata(records[0])
    expected = int(metadata.get("tensor_count", 457))
    rev = records[2]
    runs, order, trims, cand, table, json_len = search_stitch(rev, expected, expected_keys)
    shards = build_data_shards(rev, runs, table["_max_end"])
    perm, buf, score = order_shards(shards, table["tensors"], table["_max_end"], ref_weight)
    sd = {}
    for t in table["tensors"]:
        off = int(t["offset"])
        ln = int(t["length"])
        arr = np.frombuffer(buf[off:off + ln], dtype="<f2").copy()
        sd[t["key"]] = torch.from_numpy(arr).reshape(tuple(int(x) for x in t["shape"]))
    ckpt = {
        "weight": sd,
        "config": table["config"],
        "info": table.get("info", ""),
        "sr": table.get("sr", metadata.get("sr", "")),
        "f0": table.get("f0", metadata.get("f0", 1)),
        "version": table.get("version", metadata.get("version", "v2")),
    }
    if out_dir is None:
        out_dir = os.path.dirname(os.path.abspath(path))
    name = voice_name or metadata.get("model_id") or os.path.splitext(os.path.basename(path))[0]
    vdir = os.path.join(out_dir, name)
    os.makedirs(vdir, exist_ok=True)
    out_pth = os.path.join(vdir, name + ".pth")
    # Windows + 旧版 torch 对非 ASCII 路径的 PyTorchFileWriter 有兼容问题：
    # 先写到 ASCII 临时路径，再移动到最终位置。
    import shutil
    import tempfile
    fd, tmp_path = tempfile.mkstemp(suffix=".pth", prefix="ams_msvc_")
    os.close(fd)
    try:
        torch.save(ckpt, tmp_path)
        shutil.move(tmp_path, out_pth)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    report = {
        "source": os.path.abspath(path),
        "source_sha256": sha256_file(path),
        "source_size": os.path.getsize(path),
        "voice_name": name,
        "output_pth": os.path.abspath(out_pth),
        "output_size": os.path.getsize(out_pth),
        "metadata": metadata,
        "json_fragments_order": order,
        "json_trims": trims,
        "json_len": json_len,
        "data_shards": [len(s) for s in shards],
        "shard_order": list(perm),
        "shard_score": score,
        "tensor_count": len(sd),
        "max_end": table["_max_end"],
        "extra_data_bytes": len(buf) - table["_max_end"],
        "trailer_hex": trailer.hex(),
        "reference_pth": os.path.abspath(reference_pth) if reference_pth else None,
        "warnings": [],
    }
    if not os.path.exists(os.path.join(vdir, name + ".index")):
        report["warnings"].append("无同名 .index：按 00 约定标记为无检索增强（index_rate=0），不伪造")
    rp = report_path or os.path.join(vdir, "conversion.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report, os.path.abspath(rp), os.path.abspath(out_pth)


def main():
    ap = argparse.ArgumentParser(description="MSVC -> RVC v2 pth converter (AMS M0)")
    ap.add_argument("input", help="输入的 .msvc 文件")
    ap.add_argument("--out", default=None, help="输出根目录（默认输入同级）")
    ap.add_argument("--name", default=None, help="音源名称（默认 metadata.model_id 或文件名）")
    ap.add_argument("--report", default=None, help="转换报告 JSON 路径")
    ap.add_argument("--reference-pth", default=None, help="参考 RVC 小模型（同架构，用于键集校验与分片顺序）")
    ap.add_argument("--json", action="store_true", help="仅输出报告 JSON")
    args = ap.parse_args()
    report, rp, pth = convert(args.input, args.out, args.name, args.report, args.reference_pth)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("OK ->", pth)
        print("report ->", rp)
        for w in report["warnings"]:
            print("WARN:", w)


if __name__ == "__main__":
    main()
