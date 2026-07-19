# 高效自动化知识蒸馏与智能体生成管线

把任意学科教材（文本、图表、**音频**）自动转化为可本地运行、低 Token 消耗、带引用溯源的垂直领域问答智能体。基于 Ollama，消费级硬件可跑。

## 这是什么

一条端到端的「知识蒸馏」管线（此处指知识结构化提取与轻量封装，**非**模型权重蒸馏）：

```
输入源                                                    ┌─ 检索(预算控制+动态预算) → 引用锚定问答
  ├─ PDF 教材 ─► 混合路由解析 ─┐                          │      · 文本/图表 [p.X]
  │   纯文字页走文本通道；含图页调 qwen3-vl 解析图表标签   │      · 音频 [audio mm:ss]
  └─ 音频文件 ─► faster-whisper 本地转写(带时间戳) ─┤       └─ 计算器工具(白名单 AST)
                                                   ▼
                        语义分块 + 元数据 ─► bge-m3 向量化 ─► 向量库(可图文+音频混合)
```

核心特性：
- **混合路由**：纯文字页走文本通道，含图页自动走 VL（多模态）解析，把图表里的标签也变成可检索文字
- **音频 ASR**：本地 faster-whisper 转写（CPU/int8，无需云端），转写文本按时间戳分块并入同一向量库
- **Token 高效**：语义分块 + 检索预算控制，单次问答 Token ≈ 基线的 40%（实测三学科平均 48.6%，最优 38%）
- **引用锚定**：回答带 `[p.X]` 页码 / `[audio mm:ss]` 时间戳来源；无依据则 `[NO REFERENCE FOUND]`，不编造
- **动态预算**：检索命中却拒答时自动升配重答一次，把过度拒答从 20% 降至 6.7%，且仅少数题触发
- **智能体生成**：一条命令产出「定制 system prompt + 工具链(检索/计算器) + Ollama 配置 + 一键脚本」
- **持久化**：build 一次存盘，ask 多次直接加载，不重复 embedding

## 环境要求

- Windows / macOS / Linux，建议 ≥16GB RAM（有 GPU 更快；无 GPU 走 CPU 降级）
- [Python](https://www.python.org/) 3.11+
- [Ollama](https://ollama.com/)（已启动）

## 一键安装（Windows）

```cmd
cd code
setup.bat
```

脚本会自动：建虚拟环境 → 装依赖 → 拉模型（qwen3:8b / qwen3-vl:8b / bge-m3）→ 构建自定义助手模型。

### 手动安装（macOS / Linux 或想自己来）

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
ollama pull qwen3:8b && ollama pull qwen3-vl:8b && ollama pull bge-m3
ollama create distill-assistant -f Modelfile        # 可选

# 音频功能需下载 whisper 权重（阿里源，稳定）：
modelscope download --model pengzhendong/faster-whisper-small --local_dir ./models/faster-whisper-small
```

## 使用

### 1) 建库

```bash
# PDF：混合路由入库（含图页自动走 VL）
python main.py build --pdf ../data/med.pdf --vl-limit 15

# 纯文本书可跳过 VL，更快：
python main.py build --pdf ../data/cs.pdf --no-vl

# 音频：本地 ASR 转写后入库（append，可叠加到已建库上）
#   --max-seconds 只取前 N 秒（对齐 ≤5min 口径）
python main.py build --audio ../data/Starmer.mp3 --max-seconds 300

# 混合库：先 PDF（替换建库）后音频（append）→ 图文+音频同库
python main.py build --pdf ../data/med.pdf
python main.py build --audio ../data/Starmer.mp3 --max-seconds 300
```

### 2) 提问 / 连续问答

```bash
python main.py ask "What are the parts of a bacteriophage?"
python main.py chat
```

回答示例（各自正确溯源）：

```
> What are the parts of a bacteriophage?
The parts include the capsid, viral genome, sheath, and tail fibers [p.2].
[来源] p17(figure)、p18(figure)  | tokens: 575

> What did the speaker say about the Labour Government?
...
[来源] audio 00:00、audio 03:19  | tokens: 568
```

### 3) 生成智能体包

```bash
# 对已建库/PDF 生成开箱即用的智能体（含检索 + 计算器双工具）
python main.py agent --pdf ../data/med.pdf
#   产出 ./agent_<书名>/：system_prompt.txt · Modelfile · run.bat · README.md
```

进入该目录双击 `run.bat`（先 `ollama create` 专属模型，再启动对话）。对话中：
- 普通问题走**检索**（带 `[p.X]` / `[audio mm:ss]` 引用）
- 输入算式（如 `3*(4+5)`、`sqrt(144)`）走**计算器**，返回精确结果（白名单 AST，安全无注入）

### 4) 单元测试

```bash
pip install pytest
python -m pytest test_pipeline.py -v      # 44 项：计算器安全/路由/拒答判定/分块/时间戳
```

### 5) 评测脚本

```bash
python bench.py            # 三学科 Token 效率 + Hit@5
python ablation.py         # 检索预算消融
python hallucination.py    # 幻觉率 + 过度拒答实测
python dynamic_eval.py     # 动态预算：三模式对照（固定900 / 固定1800 / 动态）
```

## 参数

**build**

| 参数 | 默认 | 说明 |
|---|---|---|
| `--pdf` | — | 要入库的 PDF 路径（替换建库） |
| `--max-pages` | 120 | 最多处理前 N 页 |
| `--vl-limit` | 15 | 最多对 N 个含图页做 VL 解析 |
| `--no-vl` | 关 | 纯文本模式，跳过 VL |
| `--audio` | — | 音频路径 MP3/WAV/FLAC（转写后 append） |
| `--max-seconds` | 全长 | 仅取音频前 N 秒 |
| `--asr-model` | 自动 | faster-whisper 本地模型目录 |

**agent**：`--pdf`（给定则先建库再生成）、`--max-pages` / `--vl-limit` / `--no-vl` 同上。

向量库存到 `./vectordb`，VL 结果缓存 `./vl_cache.json`（重跑不重算）。

## 目录结构

```
Ollama_test/
├─ code/
│   main.py            主程序：build / ask / chat / agent
│   asr.py             音频本地转写（faster-whisper）
│   agent_runtime.py   智能体运行时（检索 + 计算器 · 规则路由）
│   test_pipeline.py   单元测试（pytest）
│   baseline.py / bench.py / ablation.py / hallucination.py / dynamic_eval.py   基线与评测
│   Modelfile · setup.bat · requirements.txt
│   models/            faster-whisper 本地权重（modelscope 下载）
│   vectordb/          build 生成的向量库
├─ data/   med/cs/bizlaw.pdf、Starmer.mp3、eval_books.jsonl
└─ agent_<书名>/       agent 生成的智能体包
```

## 评测结论（三学科 · 各 120 页）

| 学科 | 基线 Token | 本管线 Token | 占基线 | Hit@5 |
|---|---|---|---|---|
| CS | 1591 | 738 | 46.4% | 100% |
| 医学 | 1696 | 813 | 47.9% | 100% |
| 法学 | 1504 | 774 | 51.5% | 100% |
| 平均 | 1597 | 775 | **48.6%** | **100%** |

- **多模态**：图内信息类问题 Hit@5 由 **0%（纯文本）→ 100%（文本+VL）**
- **幻觉率**：21 道不可答探针**零编造（0%）**
- **动态预算**：过度拒答 **20% → 6.7%**
- **音频**：本地转写 en / 390s、混合库跨模态检索 + 时间戳溯源

所有指标达标（Token ≤60%、Hit@5 ≥85%、幻觉率 ≤15%）。
