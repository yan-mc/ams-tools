#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AMS MSVC -> RVC v2 .pth converter (M0.5：40k/48k 数据边界校准版)。

仅处理用户本地合法持有的 .msvc 文件；不包含 .cyd 解密，不生成 .index。
格式参考: 05_新方案探讨/01_心月格式深拆/格式.md

算法概要（容器层）：
  1. 14B 头 + 3 条 "u32 长度 + 整段字节反转 payload" 记录 + 16B trailer。
  2. 第 3 条记录反转后 = JSON 张量表碎片（5 段，可打印 run）与数据分片交错。
  3. 张量表：在 5! 片段顺序与边界裁剪空间内搜索，得到唯一可解析的 JSON（457 张量）。
  4. 数据分片：以 5 个可打印 run 之间的间隙为基准（big 分片 + 4 个小分片），
     对 (big_start_delta, tail_trim) 做小范围网格搜索，使分片总长 == 张量表 max_end；
     评分 = 参考模型同名张量 log-std 相关系数 + fp16 有限值比例。
  5. 分片逻辑顺序：参考模型 log-std 相关（120 排列），并用与已知 48k 布局一致的
     排列 (2,4,0,1,3) 作为并列时的先验。
  6. 组装 state_dict -> torch.save（ASCII 临时路径规避中文路径问题）。

用法见 README.md。
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
TOOL_VERSION = "0.3.0"
CACHE_NAME = "calibration_cache.json"
LAYOUT_PRIOR_PERM = (2, 4, 0, 1, 3)  # 48k 已由 RVC load+infer 验证的布局先验


# ---------------------------------------------------------------- 基础工具
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reference(path):
    """加载参考 RVC 模型（同架构、不同训练模型），用于键集校验与分片校准评分。"""
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


# ---------------------------------------------------------------- 张量表拼接
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

    ranges = [(4, 6), (2, 3), (2, 3), (2, 7)]
    orders = [[fi] + list(p) + [li] for p in itertools.permutations(middles)]
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
                    continue
                return runs, order, list(trims) + [last_trim], cand, obj, json_len
    raise ValueError("无法恢复张量表（拼接搜索失败）")


# ---------------------------------------------------------------- 数据分片校准
def build_shards(rev, runs, big_start_delta, tail_trim):
    """以可打印 run 间隙为基准构造 5 个数据分片（物理顺序）。

    runs 为 5 个 JSON 片段在 rev 中的位置；数据分片 = 相邻 run 之间 + 末尾后缀。
    第 3 个间隙（runs[2].end .. runs[3].start）为大分片，起点允许 ±delta；
    最后一个分片（后缀）末尾允许裁掉 tail_trim 字节（容器 padding）。
    """
    gaps = []
    for i in range(len(runs) - 1):
        gaps.append((runs[i][1], runs[i + 1][0]))
    gaps.append((runs[-1][1], len(rev)))
    shards = []
    for i, (a, b) in enumerate(gaps):
        if i == 2:
            a += big_start_delta
        if i == len(gaps) - 1:
            b -= tail_trim
        if b <= a:
            raise ValueError("数据分片非法: %r" % ((a, b),))
        shards.append(rev[a:b])
    return shards


def ref_stats(ref_weight, keys):
    out = {}
    if ref_weight is None:
        return out
    for k in keys:
        if k not in ref_weight:
            continue
        a = ref_weight[k].detach().cpu().float().numpy().astype(np.float64)
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        s = float(np.std(a))
        if s > 0:
            out[k] = float(np.log(s))
    return out


def score_buffer(buf, tensors, refs, sample_step=1):
    """返回 (finite_ratio, corr, small_ratio)。corr 为与参考模型 log-std 的相关系数。"""
    arr = np.frombuffer(buf, dtype="<f2")
    n = len(arr)
    if n == 0:
        return 0.0, -1.0, 0.0
    step = max(1, n // 100000)
    sub = arr[::step].astype(np.float32)
    finite = float(np.isfinite(sub).mean())
    xs = []
    ys = []
    for i in range(0, len(tensors), sample_step):
        t = tensors[i]
        off = int(t["offset"]) // 2
        ln = int(t["length"]) // 2
        v = arr[off:off + ln].astype(np.float32)
        if v.size < 2:
            continue
        fin = np.isfinite(v)
        if float(fin.mean()) < 0.99:
            continue
        s = float(np.std(np.where(fin, v, 0.0)))
        if not np.isfinite(s) or s <= 0:
            continue
        xs.append(np.log(s))
        ys.append(refs.get(t["key"], np.nan))
    corr = -1.0
    if refs and len(xs) >= 20:
        xs_a = np.array(xs)
        ys_a = np.array(ys)
        m = np.isfinite(ys_a)
        if int(m.sum()) >= 20:
            c = float(np.corrcoef(xs_a[m], ys_a[m])[0, 1])
            corr = c if np.isfinite(c) else -1.0
    small = float((np.abs(sub[np.isfinite(sub)]) < 100).mean()) if finite > 0 else 0.0
    return finite, corr, small


def combined_score(finite, corr, small, has_ref):
    # 有参考模型：以 log-std 相关为主（校准的核心信号），有限值比例作为硬门槛
    if has_ref:
        base = max(corr, 0.0) + 0.05 * finite + 0.01 * small
        return base if finite >= 0.999 else base - 10.0
    return 2.0 * finite + 0.5 * small


def choose_perm(shards, tensors, refs, max_end, has_ref, fixed_perm=None):
    n = len(shards)
    cands = [fixed_perm] if fixed_perm is not None else list(itertools.permutations(range(n)))
    best = None
    prior = None
    for perm in cands:
        buf = b"".join(shards[i] for i in perm)
        if len(buf) != max_end:
            continue
        finite, corr, small = score_buffer(buf, tensors, refs, sample_step=max(1, len(tensors) // 96))
        sc = combined_score(finite, corr, small, has_ref)
        row = (sc, tuple(perm), finite, corr, small)
        if best is None or sc > best[0]:
            best = row
        if fixed_perm is None and tuple(perm) == LAYOUT_PRIOR_PERM:
            prior = row
    # 布局先验：若先验排列的参考相关与最优相差 < 0.05，则采用先验（48k 已验证布局）
    if best is not None and prior is not None and has_ref:
        if prior[3] >= best[3] - 0.05:
            return prior
    return best


def calibrate(rev, runs, table, refs, has_ref, log=print):
    """网格搜索 (big_start_delta, tail_trim) 使分片总长 == max_end，再定排列。"""
    max_end = table["_max_end"]
    tensors = table["tensors"]
    base = build_shards(rev, runs, 0, 0)
    base_total = sum(len(s) for s in base)
    need = max_end - base_total  # 需要增加/减少的总字节数
    log("  base shards=%r total=%d max_end=%d diff=%+d" % ([len(s) for s in base], base_total, max_end, need))
    if abs(need) > 24:
        raise ValueError("分片总长与张量表偏差过大（%+d），拒绝猜测" % need)
    combos = []
    for delta in range(-6, 7):
        for tail in range(0, 25):
            try:
                shards = build_shards(rev, runs, delta, tail)
            except ValueError:
                continue
            if sum(len(s) for s in shards) != max_end:
                continue
            combos.append((delta, tail, shards))
    if not combos:
        raise ValueError("未找到满足总长约束的边界组合（diff=%+d）" % need)
    # 第一轮：固定布局先验排列，粗评分
    coarse = []
    for delta, tail, shards in combos:
        best = choose_perm(shards, tensors, refs, max_end, has_ref, fixed_perm=LAYOUT_PRIOR_PERM)
        if best is None:
            continue
        coarse.append((best[0], delta, tail, shards, best))
    coarse.sort(key=lambda x: -x[0])
    log("  边界候选 %d 个；粗评分前 5：%s" % (len(combos), [(round(c[0], 3), c[1], c[2]) for c in coarse[:5]]))
    # 第二轮：对粗评分前 3 个组合做完整排列搜索，取全局最优
    best_all = None
    for sc0, delta, tail, shards, _b in coarse[:3]:
        best = choose_perm(shards, tensors, refs, max_end, has_ref, fixed_perm=None)
        if best is None:
            continue
        sc, perm, finite, corr, small = best
        # 与布局先验并列时偏好先验；delta 与已验证 48k 参数（-1）接近者优先
        tie_penalty = 0.0 if tuple(perm) == LAYOUT_PRIOR_PERM else 0.02
        tie_penalty += 0.003 * abs(delta + 1)
        score = sc - tie_penalty
        if best_all is None or score > best_all[0]:
            best_all = (score, delta, tail, perm, finite, corr, small, shards)
    if best_all is None:
        raise ValueError("数据分片校准失败")
    score, delta, tail, perm, finite, corr, small, shards = best_all
    return {
        "big_start_delta": delta,
        "tail_trim": tail,
        "shards": shards,
        "shard_lengths": [len(s) for s in shards],
        "perm": list(perm),
        "score": score,
        "finite_ratio": finite,
        "ref_corr": corr,
        "small_ratio": small,
        "max_end": max_end,
        "base_total": base_total,
        "diff": need,
    }


# ---------------------------------------------------------------- 缓存
def load_cache(tool_dir):
    p = os.path.join(tool_dir, CACHE_NAME)
    if os.path.exists(p):
        try:
            return json.load(open(p, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(tool_dir, cache):
    p = os.path.join(tool_dir, CACHE_NAME)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------- 转换主流程
def convert(path, out_dir=None, voice_name=None, report_path=None, reference_pth=None,
            use_cache=True, tool_dir=None):
    import torch

    tool_dir = tool_dir or os.path.dirname(os.path.abspath(__file__))
    src_sha = sha256_file(path)
    cache = load_cache(tool_dir)
    cache_key = "%s|%s" % (src_sha, sha256_file(reference_pth) if reference_pth else "no-ref")
    cached = cache.get(cache_key) if use_cache else None

    ref_weight = None
    expected_keys = None
    refs = {}
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
    if ref_weight is not None:
        refs = ref_stats(ref_weight, [t["key"] for t in table["tensors"]])
    has_ref = len(refs) >= 50

    if cached:
        print("[cache] 使用缓存校准参数: ranges=%s perm=%s" % (cached.get("shard_ranges"), cached.get("perm")))
        if cached.get("shard_ranges"):
            shards = [rev[int(a):int(b)] for a, b in cached["shard_ranges"]]
        else:
            shards = build_shards(rev, runs, int(cached["big_start_delta"]), int(cached["tail_trim"]))
        if sum(len(s) for s in shards) != table["_max_end"]:
            print("[cache] 缓存参数与当前文件不匹配，重新校准")
            cached = None
    if not cached:
        print("[calibrate] 搜索数据分片边界 ...")
        cal = calibrate(rev, runs, table, refs, has_ref)
        shards = cal["shards"]
        cached = {
            "big_start_delta": cal["big_start_delta"],
            "tail_trim": cal["tail_trim"],
            "perm": cal["perm"],
            "score": cal["score"],
            "finite_ratio": cal["finite_ratio"],
            "ref_corr": cal["ref_corr"],
            "shard_lengths": cal["shard_lengths"],
            "max_end": cal["max_end"],
            "base_total": cal["base_total"],
            "diff": cal["diff"],
            "tool_version": TOOL_VERSION,
            "source_sha256": src_sha,
            "reference_pth": os.path.abspath(reference_pth) if reference_pth else None,
        }
        cache[cache_key] = cached
        save_cache(tool_dir, cache)

    perm = cached["perm"]
    buf = b"".join(shards[i] for i in perm)
    if len(buf) != table["_max_end"]:
        raise ValueError("分片拼接长度不符: %d vs %d" % (len(buf), table["_max_end"]))
    sd = {}
    sanitized = []
    for t in table["tensors"]:
        off = int(t["offset"])
        ln = int(t["length"])
        arr = np.frombuffer(buf[off:off + ln], dtype="<f2").copy()
        fin = np.isfinite(arr)
        huge = np.zeros(arr.shape, dtype=bool)
        if fin.any():
            # RVC 权重实测量级 < ~5；>20 视为容器哨兵/边界错位残留（M0.5 校准经验值）
            huge = np.abs(np.where(fin, arr, 0.0)) > 20.0
        if not fin.all() or huge.any():
            n_bad = int((~fin).sum())
            n_huge = int(huge.sum())
            arr[~fin] = np.float16(0.0)
            arr[huge] = np.float16(0.0)
            sanitized.append({"key": t["key"], "nonfinite": n_bad, "huge": n_huge, "offset": off})
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
        "tool": "ams-tools/msvc2pth",
        "tool_version": TOOL_VERSION,
        "source": os.path.abspath(path),
        "source_sha256": src_sha,
        "source_size": os.path.getsize(path),
        "voice_name": name,
        "output_pth": os.path.abspath(out_pth),
        "output_size": os.path.getsize(out_pth),
        "metadata": metadata,
        "json_fragments_order": order,
        "json_trims": trims,
        "json_len": json_len,
        "calibration": {
            "big_start_delta": cached["big_start_delta"],
            "tail_trim": cached["tail_trim"],
            "shard_lengths": cached["shard_lengths"],
            "shard_order": perm,
            "max_end": cached["max_end"],
            "base_total": cached["base_total"],
            "total_diff": cached["diff"],
            "score": cached["score"],
            "finite_ratio": cached["finite_ratio"],
            "ref_corr": cached["ref_corr"],
            "layout_prior": list(LAYOUT_PRIOR_PERM),
            "used_layout_prior": tuple(perm) == LAYOUT_PRIOR_PERM,
            "from_cache": bool(cached and use_cache and cache.get(cache_key) is cached),
        },
        "tensor_count": len(sd),
        "sanitized_nonfinite": sanitized,
        "sanitized_total": sum(x["nonfinite"] + x.get("huge", 0) for x in sanitized),
        "trailer_hex": trailer.hex(),
        "reference_pth": os.path.abspath(reference_pth) if reference_pth else None,
        "warnings": [],
    }
    if not os.path.exists(os.path.join(vdir, name + ".index")):
        report["warnings"].append("无同名 .index：按 00 约定标记为无检索增强（index_rate=0），不伪造")
    if sanitized:
        report["warnings"].append(
            "已把 %d 个哨兵/填充异常值（非有限或 |v|>20，出现在 %d 个张量）归零，明细见 sanitized_nonfinite"
            % (report["sanitized_total"], len(sanitized)))
    if not has_ref:
        report["warnings"].append("未提供有效参考模型：分片排列仅按 fp16 合理性评分，置信度较低")
    if cached["ref_corr"] is not None and has_ref and cached["ref_corr"] < 0.3:
        report["warnings"].append("参考相关较低（%.3f）：请用 verify_rvc.py 复核" % cached["ref_corr"])
    if cached["finite_ratio"] < 0.999:
        report["warnings"].append("fp16 有限值比例 %.4f < 0.999：疑似边界未对齐" % cached["finite_ratio"])
    rp = report_path or os.path.join(vdir, "conversion.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report, os.path.abspath(rp), os.path.abspath(out_pth)


def main():
    ap = argparse.ArgumentParser(description="MSVC -> RVC v2 pth converter (AMS M0.5)")
    ap.add_argument("input", help="输入的 .msvc 文件")
    ap.add_argument("--out", default=None, help="输出根目录（默认输入同级）")
    ap.add_argument("--name", default=None, help="音源名称（默认 metadata.model_id 或文件名）")
    ap.add_argument("--report", default=None, help="转换报告 JSON 路径")
    ap.add_argument("--reference-pth", default=None, help="参考 RVC 小模型（同架构，用于键集校验与分片校准）")
    ap.add_argument("--no-cache", action="store_true", help="忽略 calibration_cache.json 重新校准")
    ap.add_argument("--json", action="store_true", help="仅输出报告 JSON")
    args = ap.parse_args()
    report, rp, pth = convert(args.input, args.out, args.name, args.report,
                              args.reference_pth, use_cache=not args.no_cache)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("OK ->", pth)
        print("report ->", rp)
        cal = report["calibration"]
        print("calibration: delta=%s tail=%s shards=%r order=%r score=%.3f corr=%.3f finite=%.4f" %
              (cal["big_start_delta"], cal["tail_trim"], cal["shard_lengths"], cal["shard_order"],
               cal["score"], cal["ref_corr"], cal["finite_ratio"]))
        for w in report["warnings"]:
            print("WARN:", w)


if __name__ == "__main__":
    main()
