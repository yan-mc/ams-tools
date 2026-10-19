# ams-tools / msvc2pth（M0 原型）

> 把用户本地的 **MSVC/XINYU_AI** 容器转成标准 RVC v2 `.pth`（不生成 `.index`）。
> 仅用于用户自有文件的本地互操作；不包含 `.cyd` 解密，不内置/分发任何权重。

## 依赖

- Python 3.9+，numpy + torch（推荐直接用 `F:/ai歌曲/RVC20260311/runtime/python.exe`）。
- 参考模型：同架构的 RVC v2 小模型（48k 用爱莉希雅标准 pth；40k 用 manbo 标准 pth），用于键集校验 + 数据分片顺序判定。

## 用法

~~~powershell
& 'F:/ai歌曲/RVC20260311/runtime/python.exe' msvc2pth.py <input.msvc> `
    --out <输出根目录> `
    --reference-pth <同架构参考.pth> `
    [--name 音源名] [--json]
~~~

输出：`<out>/<名称>/<名称>.pth` + `conversion.json`。
不会写同名 `.index`（MSVC 只有 pth 权重）；按 00 约定标记为无检索增强。

## 验证

~~~powershell
& 'F:/ai歌曲/RVC20260311/runtime/python.exe' verify_rvc.py <输出.pth> 'F:/ai歌曲/RVC20260311'
~~~
期望：`tensor_count=457`、`unexpected=[]`、missing 全部 `enc_q.*`、`RESULT= PASS`。

## 当前状态（M0）

| 项 | 状态 |
|---|---|
| MSVC 容器解析 / 3 条 TLV / 反转 | ✅ |
| 张量表 JSON 通用拼接（5 片段顺序 + 边界裁剪搜索） | ✅ 3/3 模型 |
| 457 张量键集校验（与参考模型严格一致） | ✅ |
| 数据分片顺序判定（参考模型 log-std 相关） | ✅ 爱莉希雅 |
| 尾部 padding 自适应 | ✅ 爱莉希雅 |
| 40k 模型（法国赌神/曼波Pro）数据边界校准 | ⚠️ 待完成：原始 run 边界与数据边界需逐文件校准（当前工具会拒绝或产出不可用结果） |
| RVC 加载 + 推理 | ✅ 爱莉希雅（unexpected=0，missing=103 enc_q） |

## 已知限制

- 张量片段拼接依赖“首片段以 `{"runtime_format"` 开头、尾片段以 `}]}` 结尾、共 5 段”的实测格式；新变体需重新分析。
- 数据分片边界含污染字节，目前对 40k 两个文件尚未校准；M0 计划中的任务 M0.5 会完成。
- 非 ASCII 路径：Windows 旧版 torch 写文件有兼容问题，脚本会先写 ASCII 临时文件再移动。
