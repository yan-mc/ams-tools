# ams-tools

AMS（AI Music Studio）开发/运维工具集。

## 内容

| 工具 | 说明 |
|---|---|
| `msvc2pth/` | 心月 MSVC/XINYU_AI 容器 → 标准 RVC v2 `.pth`（不生成 index、不解密 .cyd） |

## 约束

- 不含任何音源/模型权重/密钥。
- 只处理用户本地合法持有的文件；不内置分发权重。
- Windows 优先；Python 3.9+（msvc2pth 需要 numpy+torch，可用 RVC 整合包 runtime）。

## 使用

见 `msvc2pth/README.md`。
