# AITIC 知识蒸馏与智能体生成 RAG 管线

阿里国际 **AITIC** 实训项目 · 方向 A（计算机 L2 OPT）

> 当前验收基线：**v8final**（2026-08-05）；55 本教材、6 个学科、4432 道题，标准检索与逐条审计口径均已复核。

把多学科教材（PDF / EPUB / 音频 / 图片）**结构化提取**成知识，自动生成一个能在**普通消费级电脑上离线运行**的本地问答智能体——快、省、带出处、不瞎编、一键可用。

> 「蒸馏」指知识的结构化抽取与打包，**不是**模型权重蒸馏。

---

## 核心特性

- **全本地离线**：模型、向量库、推理全在本机，零云端 API，断网可运行（隐私/保密友好）。
- **多格式输入**：PDF（含扫描/图表）、EPUB、音频（本地 ASR）、独立图片（VL 通道）。
- **带出处、不瞎编**：回答附引用锚点；检索无依据时如实拒答 `[NO REFERENCE FOUND]`，而非编造。
- **动态预算 + 相关度裁剪**：按需升配上下文预算；超预算时按相关度**整块保留**，减少残缺片段导致的误拒。
- **Windows 原生桌面版**：提供 `.msi`、`.exe` 和便携 ZIP；界面在程序内运行，不打开浏览器、不启动 Web 服务。
- **面向普通用户的安装向导**：可选择安装路径，显示安装进度，桌面快捷方式默认开启且可取消；完整包内置 Ollama 运行时，只把模型权重留给首次一键配置。
- **模型管理**：可下载和切换 Qwen3 4B/8B/14B，一键配置推荐的回答/视觉/嵌入模型，或一键导入本地 GGUF/Modelfile。
- **原生界面建库**：在“资料库”中多选 PDF / EPUB，按队列查看各书建库进度；
  EPUB 会在本机转换为带页码的可检索文本，完成后可直接问答，无需输入后端命令。
- **多知识库切换**：最近建好的知识库保留在左侧，可一键切换；“新对话”只清空聊天，不会删除知识库。

## 技术栈

Ollama · Qwen3-8B（LLM）· Qwen3-VL-8B（图文）· bge-m3（嵌入）· ChromaDB（向量库）· PySide6/Qt（原生桌面）· FastAPI（兼容 WebUI）

## 目录结构

```
code/          主管线 main.py（build/ask/chat）、评测脚本、单元测试、WebUI、Modelfile
desktop_app/   PySide6 原生界面与进程内后端适配层
packaging/     PyInstaller/WiX 构建、桌面冒烟和第三方许可提示
data/          评测集 eval_*.jsonl（Ground Truth）、智能体样例 agent_med/
deploy_pack/   部署包（Modelfile / deploy.ps1 / run.sh）
Dockerfile · docker-compose*.yml · deploy.ps1 · run.sh   部署与容器化
```

## 未入库内容（因体积，随结项部署包另行提供）

为控制仓库体积（GitHub 单文件 >100MB 直接拒绝），以下**不入库、可重建或另行提供**：

- **模型权重**（约 12GB：Qwen3-8B / bge-m3 / Qwen3-VL-8B、GGUF）→ 通过 `ollama pull` 获取；预编译 GGUF（`deploy_pack/qwen3-8b-Q4_K_M.gguf`）随部署包提供。
- **教材原文**（med.pdf / cs.pdf / bizlaw.pdf、EPUB、音频）→ 属测试数据集，随交付另行提供；评测集 Ground Truth `eval_*.jsonl` 已在仓库内。
- **向量库**（`vectordb/`）→ 运行 `build` 自动重建。

## 快速开始

```bash
# 1) 固定使用 Python 3.11，并安装依赖
py -3.11 -m venv code/.venv
code/.venv/Scripts/python -m pip install -r code/requirements.txt

# 2) 源码模式需先装 Ollama 并拉模型；完整版 MSI 已内置 Ollama 运行时
ollama pull qwen3:8b && ollama pull bge-m3 && ollama pull qwen3-vl:8b

# 3) Windows 普通用户推荐直接安装 dist/AITIC-Desktop-1.0.0-x64.msi
#    源码调试原生界面：
powershell -ExecutionPolicy Bypass -File .\start_desktop.ps1

# 4) 打开“资料库”多选 PDF / EPUB → 等待队列建库 → 激活 → 直接提问

# CLI 仍可用：python code/main.py build --pdf data/med.pdf --max-pages 120
#              python code/main.py ask "什么是革兰氏阴性菌？"

# 只做演示前检查、不启动服务
powershell -ExecutionPolicy Bypass -File .\start_webui.ps1 -CheckOnly
```

桌面安装、便携版、构建、数据位置和卸载说明见
[`DESKTOP_README.md`](DESKTOP_README.md)，模型首次配置、切换和本地导入见
[`MODEL_SETUP_GUIDE.md`](MODEL_SETUP_GUIDE.md)。旧 WebUI 仍作为兼容入口保留，但不是
Beta Plus 的默认用户界面。

## 评测结果摘要（实测）

| 指标 | 结果 |
|---|---|
| v8final 规模 | 55 本书 / 6 学科 / 4432 题 |
| 可答题严格命中 | 2017 / 2153 = **93.7%** |
| 不可答题正确拒答 | 1324 / 1374 = **96.4%** |
| 自动判定编造 | 50 / 1374 = 3.6%；逐条语义审计后确认 **45 / 1374 = 3.28%**，另 5 条为探针误判、无未裁决样本 |
| 标准 Gold-term Hit@5 | 全部 2620 / 3058 = **85.7%**；answerable 2042 / 2153 = **94.8%** |
| Token | 总量 2,003,555；中位数 378 |
| 动态升配 | 784 / 4432 = **17.7%** |
| 旧三科 Token 专项 | 基线的 46.4%–51.5%；需按最终代码复测 |

完整口径、分学科数据和限制见 [`eval/评测报告_v8final_验收候选.md`](eval/评测报告_v8final_验收候选.md)。标准检索明细见 [`eval/hit5_v8final/hit5_v8final_report.md`](eval/hit5_v8final/hit5_v8final_report.md)，50 条审计见 [`eval/fab_v8_manual_audit.md`](eval/fab_v8_manual_audit.md)。现有证据仍不能替代独立多模态准确率评测和干净环境重复部署验收。

## 交付物

① 源码仓库｜② GGUF + Modelfile + 启动脚本｜③ 六学科 55 本书 / 4432 题数据与音频专项｜④ v8final 评测报告｜⑤ 技术与部署文档

---

*AITIC 实训 · 方向 A · 2026*
