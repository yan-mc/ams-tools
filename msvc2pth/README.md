# ams-tools / msvc2pth（M0.5：40k/48k 数据边界校准与音质收尾）

> 把用户本地的 **MSVC/XINYU_AI** 容器转成标准 RVC v2 `.pth`（不生成 `.index`）。
> 仅用于用户自有文件的本地互操作；不包含 `.cyd` 解密，不内置/分发任何权重。
> 工具版本：**0.4.0**（2026-10-19，task-24 更新）

## 状态（M0.5 验收：3/3 PASS，法国赌神已收尾）

| 模型 | 容器/张量表 | RVC load | RVC infer（幅度） | 脏值/哨兵 | 结果 |
|---|---|---|---|---|---|
| 爱莉希雅（48k） | ✅ | unexpected=0，missing 全 enc_q | ✅ -0.262 ~ 0.569 | 1（footer 残留，无害） | **PASS** |
| 法国赌神（40k） | ✅ | unexpected=0，missing 全 enc_q | ✅ -0.308 ~ 0.359（不饱和） | **0**（footer 已显式归零） | **PASS** |
| 曼波Pro（48k） | ✅ | unexpected=0，missing 全 enc_q | ✅ -0.475 ~ 0.836 | 1（footer 残留，无害） | **PASS** |

产物：`07_执行计划/验证/msvc2pth_out/<名称>/<名称>.pth` + `conversion.json` + `verify.txt`。
验证报告：`07_执行计划/验证/M0_MSVC转换器验证.md`。

## 依赖

- Python 3.9+，numpy + torch（推荐 `F:/ai歌曲/RVC20260311/runtime/python.exe`，torch 2.0.0）。
- 参考模型（键集校验 + 分片边界/顺序评分）：
  - 48k：`F:/ai歌曲/RVC20260311/assets/weights/爱莉希雅.pth`
  - 40k：`F:/ai歌曲/RVC20260311/assets/weights/manbo.pth`（与法国赌神同 40k/100epoch 谱系，逐张量 std 高度一致）

## 用法

~~~powershell
# 转换（命中 calibration_cache.json 时直接复用已校准分片范围）
& 'F:/ai歌曲/RVC20260311/runtime/python.exe' msvc2pth.py <input.msvc> `
    --out <输出根目录> --reference-pth <同架构参考.pth> [--name 音源名] [--no-cache]

# 分片边界诊断（新文件校准时用）
& 'F:/ai歌曲/RVC20260311/runtime/python.exe' msvc2pth.py <input.msvc> --diagnose

# 验证（RVC load + infer）
& 'F:/ai歌曲/RVC20260311/runtime/python.exe' verify_rvc.py <输出.pth> 'F:/ai歌曲/RVC20260311'
~~~

输出：`<out>/<名称>/<名称>.pth` + `conversion.json`；不写同名 `.index`（MSVC 无检索文件，按 00 约定标记 index_rate=0）。

## 算法（0.4.0）

1. **容器**：14B 头（`MSVC`+ver+`XINYU_AI`）+ 3 条「u32 长度 + 整段字节反转 payload」+ 16B trailer。
2. **张量表**：第 3 条记录反转后 = 5 段 JSON 碎片与数据分片交错；在 5! 顺序 × 边界裁剪空间搜索，得到唯一可解析 JSON（457 张量；`length=prod(shape)*2`、offset 连续校验）。
3. **数据分片**：以 5 个可打印 run 之间的间隙为基准；对边界做小范围网格/逐段搜索：
   - 硬约束：**各分片长度为偶数**（保证 FP16 对齐）+ **总长 == 张量表 max_end**；
   - 评分：参考模型同名张量 log-std 相关 + fp16 合理性（非有限/|v|>20 的坏值数）；
   - 结果按文件 sha256 缓存 `calibration_cache.json`（绝对 `shard_ranges`、`perm`、可选 `mask_data_ranges`）。
4. **分片顺序**：120 排列评分；与已验证 48k 布局 `(2,4,0,1,3)` 接近时采用该先验。
5. **结构化 footer 归零**：容器在数据流内可能嵌入 8 字节 footer（实测 `50 52 56 31 b4 c6 00 00` → fp16 50.5/0.17/-6.70/0.0），它被计入数据长度但属于容器元数据。缓存中的 `mask_data_ranges` 会把它显式归零（`conversion.json.masked_ranges` 记录原字节），避免污染 `enc_p.proj.weight`。
6. **组装**：按张量表 offset/length 切 FP16 → state_dict → `torch.save`（ASCII 临时路径规避中文路径问题）。

### 校准参数（实测，见 calibration_cache.json）

| 模型 | 分片长度（物理序 → 逻辑序 (2,4,0,1,3)） | 参考相关 | mask | max_end |
|---|---|---|---|---|
| 爱莉希雅 | 2508, 997, 57428987, 999, 497 | 0.650 | — | 57,433,988 |
| 法国赌神 | 2508, 994, 55069690, 1000, 500 | 0.9986 | [55072690, 8] | 55,074,692 |
| 曼波Pro | 2502, 1000, 57428986, 1000, 500 | 0.657 | — | 57,433,988 |

## 40k（法国赌神）根因与修复摘要（task-24）

- 旧参数把 gap1 取成 `[4001,4995)`，与真实数据流错位 1 字节 → `enc_p.proj.weight` 尾部 1,234 字节中 200 个值变成垃圾（最大 5.87e4），是 **201 个哨兵值**的主因；另 1 个来自 8 字节 `PRV1` footer。
- 修复：gap1 改为 `[4000,4994)`（起点取 run1 末字节、终点前移 1），big 分片末尾 +1 字节（`[50876,55120566)`，长度 55,069,690 偶），并显式归零 footer 8 字节。
- 结果：脏值 201 → **0**；infer 幅度 0.316~1.0（饱和）→ **-0.308~0.359（正常）**；`enc_p.proj.weight` std 14.25 → **0.06739**（参考 manbo 0.06772），|max| 1.646（参考 1.63）。

## 已知限制

- 48k 两个模型仍各有 **1 个** footer 残留值被归零（`emb_g.weight` / `enc_p.proj.weight`，无害，infer 幅度正常）；如需同样清零，可对 48k 做同样 footer mask（低优先级）。
- 数据分片边界校准依赖参考模型；无参考时仅按 fp16 合理性评分，置信度低（会给出 WARN）。
- 张量碎片拼接依赖「5 段、首段 `{"runtime_format"`、尾段 `}]}`」实测格式；新变体会明确报错，不猜测。
- 不生成 `.index`；如用户另有同名 index 可手动放入 `voices/<名称>/`。

## 文件

| 文件 | 说明 |
|---|---|
| `msvc2pth.py` | 转换器（容器解析/张量表拼接/边界校准/分片排序/footer 归零/state_dict 写出） |
| `verify_rvc.py` | RVC v2 加载+推理验证（unexpected=0、missing 全 enc_q、infer 有限） |
| `calibration_cache.json` | 每文件校准缓存：`shard_ranges`/`perm`/`mask_data_ranges`/评分/依据 |
| `build_cache.py` | 生成/更新校准缓存（记录已校准文件参数与依据） |
