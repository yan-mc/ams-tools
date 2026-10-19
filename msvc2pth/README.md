# ams-tools / msvc2pth（M0.5：40k/48k 数据边界校准版）

> 把用户本地的 **MSVC/XINYU_AI** 容器转成标准 RVC v2 `.pth`（不生成 `.index`）。
> 仅用于用户自有文件的本地互操作；不包含 `.cyd` 解密，不内置/分发任何权重。
> 工具版本：0.3.0（2026-10-19，task-18）

## 状态（M0.5 验收：3/3 PASS）

| 模型 | 容器/张量表 | 457 张量键集 | RVC load | RVC infer | 结果 |
|---|---|---|---|---|---|
| 爱莉希雅（48k） | ✅ | ✅ | unexpected=0，missing 全 enc_q | ✅ 有限（-0.262 ~ 0.569） | **PASS** |
| 法国赌神（40k） | ✅ | ✅ | unexpected=0，missing 全 enc_q | ✅ 有限（0.316 ~ 1.0，见下方保真度说明） | **PASS** |
| 曼波Pro（48k） | ✅ | ✅ | unexpected=0，missing 全 enc_q | ✅ 有限（-0.475 ~ 0.836） | **PASS** |

产物：`07_执行计划/验证/msvc2pth_out/<名称>/<名称>.pth` + `conversion.json` + `verify.txt`。
验证日志：`07_执行计划/验证/M0_MSVC转换器验证.md`。

## 依赖

- Python 3.9+，numpy + torch（推荐直接用 `F:/ai歌曲/RVC20260311/runtime/python.exe`，torch 2.0.0）。
- 参考模型（用于键集校验 + 数据分片边界/顺序评分）：
  - 48k：`F:/ai歌曲/RVC20260311/assets/weights/爱莉希雅.pth`（arch 一致，不同训练模型）
  - 40k：`F:/ai歌曲/RVC20260311/assets/weights/manbo.pth`

## 用法

~~~powershell
# 单文件转换
& 'F:/ai歌曲/RVC20260311/runtime/python.exe' msvc2pth.py <input.msvc> `
    --out <输出根目录> --reference-pth <同架构参考.pth> [--name 音源名] [--no-cache]

# 验证（RVC load + infer）
& 'F:/ai歌曲/RVC20260311/runtime/python.exe' verify_rvc.py <输出.pth> 'F:/ai歌曲/RVC20260311'
~~~

输出：`<out>/<名称>/<名称>.pth` + `conversion.json`；不写同名 `.index`（MSVC 只有 pth 权重，按 00 约定标记无检索增强）。

## 算法（0.3.0）

1. **容器**：14B 头（`MSVC`+ver+`XINYU_AI`）+ 3 条「u32 长度 + 整段字节反转 payload」+ 16B trailer。
2. **张量表**：第 3 条记录反转后 = 5 段 JSON 碎片 + 数据分片交错；在 5! 片段顺序 × 边界裁剪（首/尾 0-6、中间 0-2）空间内搜索，得到唯一可解析 JSON（457 张量，`length=prod(shape)*2`、offset 递增全部校验）。
3. **数据分片**：以 5 个可打印 run 之间的间隙为基准（大分片 + 4 个小分片），对边界做小范围网格搜索：
   - 硬约束：分片总长 == 张量表 `max_end`；
   - 评分：参考模型同名张量 **log-std 相关系数** + fp16 有限值比例；
   - 结果按文件 sha256 缓存到 `calibration_cache.json`（含绝对 `shard_ranges`），`--no-cache` 可强制重算。
4. **分片顺序**：120 排列按 log-std 相关评分；与已验证 48k 布局 `(2,4,0,1,3)` 相差 <0.05 时采用该先验（`used_layout_prior=true`）。
5. **组装**：按张量表 offset/length 切 FP16 → state_dict → `torch.save`（ASCII 临时路径规避中文路径问题）。
6. **哨兵修复（M0.5 新增）**：容器在边界处存在少量 fp16 哨兵/填充值（NaN/Inf 或 |v|>20）。转换后统计并归零，数量与位置写入 `conversion.json.sanitized_nonfinite`，并给出 WARN。

### 校准参数（3 个文件实测，见 calibration_cache.json）

| 模型 | 分片长度（物理序） | 逻辑顺序 | 参考相关 | 哨兵归零数 | max_end |
|---|---|---|---|---|---|
| 爱莉希雅 | 2508, 997, 57428987, 999, 497 | (2,4,0,1,3) | 0.650 | 1 | 57,433,988 |
| 法国赌神 | 2508, 994, 55069688, 1000, 502 | (2,4,0,1,3) | 0.9995 | 201 | 55,074,692 |
| 曼波Pro | 2502, 1000, 57428986, 1000, 500 | (2,4,0,1,3) | 0.657 | 1 | 57,433,988 |

## 已知限制（务必阅读）

- **法国赌神保真度低于另两个模型**：数据分片边界仍有少量无法定位的哨兵字节（201 个值被归零），infer 输出有限但偏饱和（0.316~1.0）。结构/加载/推理链路已通过验收，但作为最终音色使用前建议进一步动态校验（例如在 RVC 运行时对比同一输入下 manbo 参考模型与转换模型的中间张量）。
- 张量碎片拼接依赖「5 段、首段以 `{"runtime_format"` 开头、尾段以 `}]}` 结尾」的实测格式；新变体需重新分析（工具会明确报错，不猜测）。
- 数据分片边界校准依赖参考模型；无参考时置信度低（会给出 WARN）。
- 不生成 `.index`（MSVC 不含检索文件）；如用户另有同名 index 可手动放入 `voices/<名称>/`。

## 文件

| 文件 | 说明 |
|---|---|
| `msvc2pth.py` | 转换器（容器解析 / 张量表拼接 / 边界校准 / 分片排序 / state_dict 写出 / 哨兵修复） |
| `verify_rvc.py` | RVC v2 加载+推理验证（unexpected=0、missing 全 enc_q、infer 有限） |
| `calibration_cache.json` | 每文件（源 sha256 + 参考 sha256）校准缓存：`shard_ranges`/perm/评分/来源 |
| `build_cache.py` | 生成/更新上述缓存（记录 3 个已校准文件的参数与依据） |
