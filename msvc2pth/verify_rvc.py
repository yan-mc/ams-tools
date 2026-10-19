#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""验证转换后的 .pth 可被 RVC v2 加载并推理。
用法: RVC/runtime/python.exe verify_rvc.py <pth> <RVC根目录>
"""
import os
import shutil
import sys
import tempfile

import torch


def main():
    if len(sys.argv) < 3:
        print("usage: verify_rvc.py <pth> <RVC_ROOT>")
        return 2
    pth_path = os.path.abspath(sys.argv[1])
    rvc_root = os.path.abspath(sys.argv[2])
    # 非 ASCII 路径兼容：复制到 ASCII 临时文件再 torch.load
    fd, tmp = tempfile.mkstemp(suffix=".pth", prefix="ams_verify_")
    os.close(fd)
    try:
        shutil.copyfile(pth_path, tmp)
        ckpt = torch.load(tmp, map_location="cpu")
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    sd = ckpt["weight"]
    cfg = list(ckpt["config"])
    info = ckpt.get("info")
    sr = ckpt.get("sr")
    sys.path.insert(0, rvc_root)
    from infer.lib.infer_pack.models import SynthesizerTrnMs768NSFsid
    net = SynthesizerTrnMs768NSFsid(*cfg, is_half=False)
    ret = net.load_state_dict(sd, strict=False)
    print("tensor_count=", len(sd))
    print("info=", info, "sr=", sr, "version=", ckpt.get("version"), "f0=", ckpt.get("f0"))
    print("unexpected=", list(ret.unexpected_keys))
    print("missing_count=", len(ret.missing_keys))
    print("missing_all_enc_q=", all(k.startswith("enc_q.") for k in ret.missing_keys))
    net.eval().float()
    T = 40
    torch.manual_seed(0)
    phone = torch.randn(1, T, 768)
    plen = torch.LongTensor([T])
    pitch = torch.randint(1, 255, (1, T))
    pitchf = torch.rand(1, T) * 200
    sid = torch.LongTensor([0])
    with torch.no_grad():
        audio = net.infer(phone, plen, pitch, pitchf, sid)
    if isinstance(audio, tuple):
        audio = audio[0]
    print("infer_shape=", tuple(audio.shape))
    print("infer_finite=", bool(torch.isfinite(audio).all()))
    print("infer_min=", float(audio.min()), "max=", float(audio.max()))
    ok = (len(ret.unexpected_keys) == 0 and len(ret.missing_keys) > 0
          and all(k.startswith("enc_q.") for k in ret.missing_keys)
          and bool(torch.isfinite(audio).all()))
    print("RESULT=", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
