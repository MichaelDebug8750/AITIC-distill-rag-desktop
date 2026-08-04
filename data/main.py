"""
main.py — 知识蒸馏管线（整合版 · 命令行入口）
================================================================
把之前散落的"混合路由入库 + 检索问答"整合成一个可复用主程序。
核心能力：
  · 混合路由：纯文字页走文本通道，含图页自动走 VL（qwen3-vl）解析
  · 多格式入库：PDF（混合路由）/ 音频（ASR）/ EPUB（转文本）/ 独立图片（VL），后三者 append 成混合库
  · 持久化向量库：build 一次存盘，ask 时直接加载，不重复 embedding
  · 引用锚定：回答带 [p.X]/[audio mm:ss]/[ch:标题]/[image:名] 溯源，无依据时输出 [NO REFERENCE FOUND]
  · 动态上下文裁剪：按相关度整块保留（RELEVANCE_TRIM），预算内优先保住最相关的完整块
  · VL 缓存 + 失败重试：解析过的页不重算，网络抖动自动重试

依赖：pip install -r requirements.txt
模型：ollama pull qwen3:8b  /  ollama pull qwen3-vl:8b  /  ollama pull bge-m3

用法：
  # 1) 建库（对一本书做混合路由入库，结果存到 ./vectordb）
  python main.py build --pdf med.pdf

  # 只取前 60 页、最多 10 个含图页走 VL；纯文本模式加 --no-vl
  python main.py build --pdf med.pdf --max-pages 60 --vl-limit 10
  python main.py build --pdf cs.pdf --no-vl

  # 2) 提问（加载已建好的库）
  python main.py ask "What are the parts of a bacteriophage?"

  # 3) 进入连续问答（输入 exit 退出）
  python main.py chat
"""

import argparse, base64, json, os, re, sys, time
import fitz
import chromadb

try:
    import ollama
except ImportError:
    print("缺少 ollama 库，请先 pip install -r requirements.txt"); sys.exit(1)

# ----------------------------- 配置 -----------------------------
LLM_MODEL = "qwen3:8b"
VL_MODEL = "qwen3-vl:8b"
EMBED_MODEL = "bge-m3"

CHUNK_TARGET, CHUNK_MAX = 450, 650
TOP_K, CONTEXT_BUDGET = 8, 900          # 消融得到的最优预算
NUM_PREDICT, TEMPERATURE = 300, 0.0
BUDGET_ESCALATED = 1800                  # 动态预算：首答拒答且检索有命中时升配重答一次
DYNAMIC_BUDGET = True                    # 关掉则恒用 900（用于对照评测）
# 升配闸门：首答拒答后，若最优检索块的距离大于此值，判定为"库里真没有"，不再升配。
# None = 关闭闸门（旧行为，全部升配）。数值需先用 calib_gate2.py 标定。
# 1.1762 = v3 终版跨学科标定值，全量 4432 题验证：升配 -21.3%、token -5.4%、可答持平。
# 【2026-07-30 改】默认值由 None 改为 1.1762：v3 交付口径的全部数字都是带闸门跑出来的，
# 默认 None 会让任何忘传 --set-gate 的运行静默退回非交付配置（v6chk 升配率 40.8% 即此因）。
# 做「无闸门」对照时显式传 --set-gate none，不要靠默认值。
ESCALATE_SIM_GATE = 1.1762

# 按学科分档的升配闸门。实测（250 题教材 + 235 题文学，同代码仅换 gate）：
#   教材（Business/CS/Medicine）0.96 vs 1.1762：升配 -69%、token -5%、准确度持平
#   文学（Moby-Dick/Sherlock/Jane Eyre）0.96 vs 1.1762：可答 -5.9pp，8 道由作答变拒答
# 机制：叙事文本的人物/情节问句与原文措辞差异大，检索距离天然偏高，
#       低闸门会把本来答得出来的题误判为"库里没有"。
# 未列出的学科回退到 ESCALATE_SIM_GATE（保守值），因为闸门定低的代价是漏答（伤准确度），
# 定高的代价只是多花 token——评委优先级里准确度远大于 token。
# 【2026-08-01 修订】全量 4432 题实测后的分档表。逐题翻转（v7full -> v8gate）：
#   Psychology  净 +2  因闸门漏答 0 道  升配 160->47   <- 低闸门最干净
#   Business    净 +2  因闸门漏答 3 道  升配 254->87
#   Literature  净  0  （本就是高闸门）  升配 192->195
#   CS          净 -1  因闸门漏答 1 道  升配 167->46
#   Medicine    净 -3  因闸门漏答 6 道  升配 265->90    <- 提回高闸门
#   Law         净 -5  因闸门漏答 5 道  升配 150->41    <- 提回高闸门
# Law/Medicine 的「How is X discussed in this book」型提问检索距离偏大，
# 行为更接近叙事文本而非术语型教材，低闸门会把答得出来的题误判为库里没有：
#   "How is surety discussed in this book?"  1.1762 下详述 suretyship，0.96 下 [NO REFERENCE FOUND]
# 取舍依据：评委优先级为准确度 >> 时间 >> token，故宁可多花 token 也不漏答。
ESCALATE_GATE_BY_SUBJECT = {
    "Business": 0.96,          # A/B: 净 +2
    "Computer Science": 0.96,  # A/B: 净 -1（噪声内）
    "Psychology": 0.96,        # A/B: 净 +2，零漏答
    "Medicine": 1.1762,        # A/B: 0.96 下净 -3、漏答 6 道 -> 保守档
    "Law": 1.1762,             # A/B: 0.96 下净 -5、漏答 5 道 -> 保守档
    "Literature": 1.1762,      # A/B: 0.96 下可答 -5.9pp
}
# 关掉分档、全局用 ESCALATE_SIM_GATE：DISTILL_GATE_BY_SUBJECT=0
GATE_BY_SUBJECT = os.environ.get("DISTILL_GATE_BY_SUBJECT", "1").strip() not in ("0", "off", "false", "no")
try:
    _m = os.environ.get("DISTILL_GATE_MAP", "").strip()
    if _m:
        ESCALATE_GATE_BY_SUBJECT.update(json.loads(_m))
except Exception:
    sys.stderr.write("[警告] DISTILL_GATE_MAP 解析失败，沿用内置分档表\n")


def resolve_gate():
    """取当前库应使用的升配闸门。学科由建库时写入 build_manifest.json，
       所以调用方不需要知道学科——库自己记得自己装的是什么。"""
    if not GATE_BY_SUBJECT:
        return ESCALATE_SIM_GATE
    subj = (read_manifest() or {}).get("subject")
    if subj:
        wanted = str(subj).strip().casefold()
        for name, gate in ESCALATE_GATE_BY_SUBJECT.items():
            if name.casefold() == wanted:
                return gate
    return ESCALATE_SIM_GATE
RELEVANCE_TRIM = True                    # 动态裁剪：按相关度整块保留（关掉=旧的按检索序字符截断，用于对照）
VL_DPI = 150
VL_RETRY = 3

DB_PATH = "./vectordb"
COLLECTION = "knowledge_base"
VL_CACHE = "./vl_cache.json"

HEADING_RE = re.compile(r"^\s*(\d+(\.\d+)*[\.\)]|[A-Z][A-Z ]{3,}$|Chapter\s+\d+|CHAPTER\s+\d+)")

# ---------- PROMPT 变体 ----------
# 背景（2026-07-30）：ollama 服务端 0.31.1 -> 0.32.3 后，qwen3:8b 对"完形填空/关键词碎片"
# 式问题不再自发写正文，只回一个 [p.XXX] 标签（think=False 下 eval_count=7）。
# 检索、打包、参数全部正常，v3full 存档答案证明模型当年是会写正文的。
# 补一句显式指令即可恢复，V1 的输出与 v3full 存档逐字节相同。
# 用环境变量切换，便于 A/B：DISTILL_PROMPT_VARIANT=V0|V1|V2|V3
_PROMPT_V0 = """Answer the question using ONLY the material below. Each block starts with a source tag in square brackets, for example {tag_example}.
When you cite, copy the tag of the block you actually used EXACTLY as shown above that block. Do NOT invent page numbers and do NOT change the tag format.
If there is no basis in the material, answer exactly "[NO REFERENCE FOUND]".

Material:
{context}

Question: {question}
Answer:"""

# V1：题面之后加一句，位置离生成最近、最强硬。副作用是把正常答案也压短了（实测 -58%）。
_PROMPT_V1 = _PROMPT_V0.replace(
    "Question: {question}\nAnswer:",
    "Question: {question}\nWrite the answer as a complete sentence, then the tag. "
    "Never answer with a tag alone.\nAnswer:")

# V2：接在拒答句之后，说明"标签本身不算答案"。
_PROMPT_V2 = _PROMPT_V0.replace(
    'If there is no basis in the material, answer exactly "[NO REFERENCE FOUND]".',
    'If there is no basis in the material, answer exactly "[NO REFERENCE FOUND]".\n'
    'Otherwise your answer must contain the actual information in your own words BEFORE the tag. '
    'A tag by itself is not an answer.')

# V3：告诉模型题面可能是填空句或关键词碎片。修复效果与 V1 相同，
#     但对正常题的扰动最小（正文 608->511 vs V1 的 608->253）。
_PROMPT_V3 = _PROMPT_V0.replace(
    "Question: {question}\nAnswer:",
    "Question: {question}\n"
    "The question may be a sentence with a missing term, or just keywords. "
    "In either case, state the relevant fact from the material in full, then cite the tag.\n"
    "Answer:")

# V5：V3 + 反向约束。目标是压低 V3 带来的 +15 道真幻觉。
# 观察到的失败形态：模型抓住"语义邻近但并非所问"的内容硬套——
#   "oral stage"(口欲期) 撞上最高法院的 oral arguments
#   "latency period"(潜伏期) 撞上磁盘 latency
#   "probable cause" / "public opinion" / "positive statement" 同理
# 所以约束的着力点不是"要不要答"，而是"材料是否真的覆盖了所问的那个东西"。
# 【风险】这一句可能把 V3 好不容易压下去的过度拒答（7.7%->3.0%）顶回来，
#         因此采纳判据必须双向：过度拒答回升 <=1pp 且真幻觉下降 >=5 道。
_PROMPT_V5 = _PROMPT_V3.replace(
    'If there is no basis in the material, answer exactly "[NO REFERENCE FOUND]".',
    'If there is no basis in the material, answer exactly "[NO REFERENCE FOUND]".\n'
    'The material must actually cover the specific thing being asked about. '
    'Content that merely sounds similar, shares a word, or is loosely related is NOT a basis '
    '— in that case answer "[NO REFERENCE FOUND]" rather than stretching the material to fit.')

PROMPT_VARIANTS = {"V0": _PROMPT_V0, "V1": _PROMPT_V1, "V2": _PROMPT_V2,
                   "V3": _PROMPT_V3, "V5": _PROMPT_V5}
# 【2026-07-30 改】默认由 V0 改为 V3。
# 依据：ollama 服务端 0.31.1 -> 0.32.3 后，V0（原版 PROMPT）在本环境下已损坏——
#       模糊题 48.2% -> 17.9%，模型对完形填空式问题只回引用标签不写正文。
#       V3 经 3 本 250 题实测，可答 110/119=92.4%、模糊 27/56=48.2%，与 v3full 基线计数逐项相同。
#       V0 保留在变体表里，用途是复现 ollama 0.31 时代的历史结果。
# 注意：这个默认值只有 3 本书的证据，全量 4432 题验证前不要当作最终交付口径。
PROMPT_VARIANT = os.environ.get("DISTILL_PROMPT_VARIANT", "V3").strip().upper()
if PROMPT_VARIANT not in PROMPT_VARIANTS:
    sys.stderr.write("[警告] 未知 PROMPT 变体 %r，回落 V0\n" % PROMPT_VARIANT)
    PROMPT_VARIANT = "V0"
# 默认仍是 V0（现网口径）。改默认值必须先有 250 题实测支撑。
PROMPT = PROMPT_VARIANTS[PROMPT_VARIANT]


# ----------------------------- 工具函数 -----------------------------
def split_sentences(text):
    return [x.strip() for x in re.split(r"(?<=[.!?。！？；\n])", text) if x.strip()]


def semantic_chunks(text):
    """语义分块：识别标题做前缀，按句子边界凑成 ~CHUNK_TARGET 的块。"""
    chunks, buf, blen, heading = [], [], 0, ""

    def flush():
        nonlocal buf, blen
        if buf:
            chunks.append((heading + " " + "".join(buf)).strip())
            buf, blen = [], 0

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        if HEADING_RE.match(line) and len(line) < 60:
            flush(); heading = line; continue
        for s in split_sentences(line):
            if blen + len(s) > CHUNK_MAX and buf:
                flush()
            buf.append(s); blen += len(s)
            if blen >= CHUNK_TARGET:
                flush()
    flush()
    return chunks


def _embed_http(texts):
    """直接 HTTP 调 Ollama /api/embed（等价 curl），绕过 ollama 库版本不兼容。"""
    import urllib.request
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    if not host.startswith("http"):
        host = "http://" + host
    body = json.dumps({"model": EMBED_MODEL, "input": texts}).encode("utf-8")
    req = urllib.request.Request(host.rstrip("/") + "/api/embed", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode("utf-8"))["embeddings"]


def embed(texts):
    # 优先用 ollama 库；库与服务端版本不匹配（如 502）时自动降级到 HTTP 直连
    try:
        return ollama.embed(model=EMBED_MODEL, input=texts)["embeddings"]
    except (AttributeError, KeyError, TypeError):
        try:
            return [ollama.embeddings(model=EMBED_MODEL, prompt=t)["embedding"] for t in texts]
        except Exception:
            return _embed_http(texts)
    except Exception:
        return _embed_http(texts)


def _ollama_host():
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
    return ("http://" + host if not host.startswith("http") else host).rstrip("/")



def _strip_think(text):  # think_off_patch
    """剥离 <think>...</think> 思考块；若剩余为空则返回拒答标记。"""
    import re as _re
    if text is None:
        return "[NO REFERENCE FOUND]"
    t = _re.sub(r"<think>.*?</think>", "", str(text), flags=_re.S | _re.I)
    t = _re.sub(r"</?think>", "", t, flags=_re.I)
    t = t.strip()
    return t if t else "[NO REFERENCE FOUND]"


def _post_json(path, payload):
    import urllib.request
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(_ollama_host() + path, data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode("utf-8"))


# ---- 生成调用统计（可观测性）----
# 踩坑#12 —— /api/chat 与 /api/generate 是两种不同的调用模式，前者会让模型编造页码、
# 答案形态也不同。下面的 except 分支是静默降级，一旦触发，整轮评测的数字其实来自
# 另一种调用模式，而产物里看不出任何痕迹。这里把它计数并暴露到指纹里。
GEN_STATS = {"total": 0, "fallback": 0, "first_error": None}


def _generate(model, prompt, system=None, options=None):
    """生成：优先 ollama 库；库不兼容（502 等）时降级 HTTP /api/generate。
       返回对象统一支持 out["response"] 与 out.get(key, default)。"""
    GEN_STATS["total"] += 1
    try:
        kw = {"model": model, "prompt": prompt, "think": False}   # think=False：避免推理块被 _strip_think 剥空后误判拒答
        if system is not None:
            kw["system"] = system
        if options is not None:
            kw["options"] = options
        _r = ollama.generate(**kw)
        try:
            _r["response"] = _strip_think(_r.get("response", ""))
        except Exception:
            pass
        return _r
    except Exception as _e:
        # Ollama 0.31+ 的 /api/generate 对 prompt 收严会返 400，改走 /api/chat
        GEN_STATS["fallback"] += 1
        if GEN_STATS["first_error"] is None:
            GEN_STATS["first_error"] = ("%s: %s" % (type(_e).__name__, _e))[:200]
            sys.stderr.write(
                "\n[警告] ollama.generate 调用失败，已降级到 /api/chat。\n"
                "        这是踩坑#12 记录的\"会改变答案形态/编造页码\"的调用模式，\n"
                "        本次运行的所有指标都不能直接与 /api/generate 的历史结果对比。\n"
                "        首次异常：%s\n\n" % GEN_STATS["first_error"])
        msgs = []
        if system is not None:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})
        payload = {"model": model, "messages": msgs, "stream": False, "think": False}
        if options is not None:
            payload["options"] = options
        out = _post_json("/api/chat", payload)
        text = _strip_think(out.get("message", {}).get("content", ""))
        # 统一成 generate 的返回形状，兼容上游 out["response"] / out.get(key)
        out["response"] = text
        return out


def _chat_vl(model, prompt, b64):
    """VL 多模态对话：优先 ollama 库；降级 HTTP /api/chat。返回 message.content 文本。"""
    try:
        out = ollama.chat(model=model, messages=[{"role": "user", "content": prompt, "images": [b64]}])
        return out["message"]["content"]
    except Exception:
        out = _post_json("/api/chat", {"model": model, "stream": False,
                                       "messages": [{"role": "user", "content": prompt, "images": [b64]}]})
        return out["message"]["content"]


def vl_parse(img_bytes):
    """对一页图调 VL，提取图表里的文字标签 + 描述；失败自动重试。"""
    b64 = base64.b64encode(img_bytes).decode()
    p = ("This is a textbook page. Extract ALL text labels from any figures, diagrams, or "
         "tables (e.g. component names, scale bars, axis labels). List every label, then "
         "briefly describe what each figure shows.")
    for attempt in range(VL_RETRY):
        try:
            return _chat_vl(VL_MODEL, p, b64)
        except Exception:
            print(" [VL重试%d]" % (attempt + 1), end="", flush=True); time.sleep(3)
    return ""


def load_vl_cache():
    if os.path.exists(VL_CACHE):
        try:
            return json.load(open(VL_CACHE, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_vl_cache(cache):
    """把 VL 描述缓存落盘。断点续传的关键：每处理完一页就调一次，中断也不丢已处理页。"""
    json.dump(cache, open(VL_CACHE, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


# ----------------------------- 建库 -----------------------------
# VL 转写开关：DISTILL_VL_PROSE=0 关闭（做 A/B 对照用）。默认开启。
# 注意：该开关已计入 chunking_fingerprint()，开关不同的两个库指纹不同，
#       不会出现"指纹相同但库内容不同"的情况。
VL_PROSE = os.environ.get("DISTILL_VL_PROSE", "1").strip() not in ("0", "off", "false", "no")


def _vl_to_prose(vtxt):
    """把 VL 输出的编号标签列表转写成自然句子后再入库。

    动机：VL 的输出形如
        ### Extracted Text Labels from Figure
        1. **Anterior fontanelle**
        2. **Frontal bone**
    这种「数字. 开头」的行会被 semantic_chunks 的 HEADING_RE 判为标题而整行丢弃
    （实测 A&P 的 VL 输出有 35.5% 的行因此进不了检索空间，图内标签全部丢失）。

    曾尝试改 HEADING_RE 从根上修，但该分块链路为全部语料共用，回归验证显示
    可答 -4.2pp、模糊 -28.6pp（3 本 250 题），代价过大，遂改为只在 VL 侧处理。

    转写后：标签合并成一个句子，既不会被判为标题，也比裸名词列表更利于语义检索。
    """
    if not vtxt or not VL_PROSE:
        return vtxt
    out, labels = [], []

    def clean(x):
        return re.sub(r"\*+", "", x).strip().strip(" -–—:")

    def flush_labels():
        if labels:
            out.append("Labels shown in this figure: " + "; ".join(labels) + ".")
            labels.clear()

    for line in vtxt.split("\n"):
        raw = line.rstrip()
        t = raw.strip()
        if not t:
            continue
        m = re.match(r"^\s*\d+[\.\)]\s+(.*)$", t)
        if m:                                   # 编号列表项 → 收集为标签
            item = clean(m.group(1))
            if item:
                labels.append(item)
            continue
        # 续行：缩进行 / *Description:* 这类行，是上一个标签的说明，不是新段落。
        # 不做这一步的话，Gray's 那种「每个编号项后跟一行说明」的格式会让 flush_labels()
        # 逐条触发，退化成 N 句「Labels shown in this figure: X.」而非设计的一句合并，
        # 且图号（Fig. 1.4）失去描述后作为检索目标毫无意义。实测该页 0/6 存活。
        if labels and (raw[:1] in " \t" or re.match(r"^\*\s*(description|desc|caption|note)\b", t, re.I)):
            extra = clean(re.sub(r"^\*\s*(description|desc|caption|note)\s*:?\s*\**", "", t, flags=re.I))
            if extra:
                labels[-1] = "%s — %s" % (labels[-1], extra)
            continue
        flush_labels()                          # 真正的非列表行：先把攒下的标签成句
        out.append(re.sub(r"^#+\s*", "", t))   # 去掉 markdown 标题井号
    flush_labels()
    return "\n".join(out)


# ==================== 库指纹与配置自证 ====================
# 背景：分块逻辑的改动活在\"向量库\"里，不活在代码里。回滚了代码但复用旧库，
#       跑出来的就是旧代码的结果，而产物里没有任何痕迹可以分辨（v6chk 即此坑）。
#       这里给库打指纹，让每次评测都能自证\"这库是哪版代码建的、跑的是哪套配置\"。
MANIFEST_NAME = "build_manifest.json"
# 环境变量 DISTILL_STRICT_LIB=1 时，库指纹不匹配直接中止，而不是仅告警
STRICT_LIB = os.environ.get("DISTILL_STRICT_LIB", "").strip().lower() in ("1", "true", "yes")
# 环境变量 DISTILL_QUIET_LIB=1 时，指纹不匹配只打一行短标记（父进程按行计数用），
# 不打整块告警——评测里 ask 是逐题起子进程，否则会刷 N 遍同样的 12 行。
QUIET_LIB = os.environ.get("DISTILL_QUIET_LIB", "").strip().lower() in ("1", "true", "yes")
STALE_MARK = "[STALE-LIBRARY]"   # 父进程识别用的固定标记，别改


def _sha(s, n=12):
    import hashlib
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()[:n]


def _code_sha():
    """当前 main.py 文件本身的哈希——用来识别\"跑的是 code\\ 还是 data\\ 那一份\"。"""
    try:
        with open(os.path.abspath(__file__), "rb") as f:
            import hashlib
            return hashlib.sha256(f.read()).hexdigest()[:12]
    except Exception:
        return "unknown"


def chunking_fingerprint():
    """分块指纹：凡是影响\"文本如何切块入库\"的东西变了，这个值就变，否则不变。
       覆盖 HEADING_RE / CHUNK_TARGET / CHUNK_MAX / EMBED_MODEL /
       semantic_chunks 与 _vl_to_prose 的源码。
       用途：证明某个向量库是 v3 原版还是修复①/修复② 建的。"""
    import inspect
    try:
        src = inspect.getsource(semantic_chunks) + inspect.getsource(_vl_to_prose)
    except Exception:
        src = "<source-unavailable>"
    return _sha("|".join([HEADING_RE.pattern, str(CHUNK_TARGET), str(CHUNK_MAX),
                          EMBED_MODEL, _sha(src, 64), "vlprose=%d" % int(VL_PROSE)]))


_ENV_FP_CACHE = {}


def env_fingerprint():
    """外部依赖版本指纹。
       这是 v6 排查里代价最大的一课：代码和数据产物都自证了，根因却在这儿——
       ollama 服务端 0.31.1 -> 0.32.3 改变了 think=False 下的生成行为。
       环境不是背景，是变量。"""
    if _ENV_FP_CACHE:
        return _ENV_FP_CACHE
    out = {"ollama_server": None, "ollama_py": None,
           "model_digest": None, "model_digest_source": None}
    import urllib.request
    try:
        with urllib.request.urlopen(_ollama_host() + "/api/version", timeout=5) as r:
            out["ollama_server"] = json.loads(r.read().decode("utf-8")).get("version")
    except Exception as e:
        out["ollama_server"] = "probe-failed: %s" % type(e).__name__
    try:
        try:
            from importlib.metadata import version as _v
        except ImportError:
            from importlib_metadata import version as _v
        out["ollama_py"] = _v("ollama")
    except Exception:
        out["ollama_py"] = "unknown"
    try:
        body = json.dumps({"model": LLM_MODEL}).encode("utf-8")
        req = urllib.request.Request(_ollama_host() + "/api/show", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            info = json.loads(r.read().decode("utf-8"))
        det = info.get("details", {}) or {}
        digest = info.get("digest") or ""
        if digest:
            out["model_digest_source"] = "api/show"
        else:
            # 新版 /api/show 通常不返回 digest；/api/tags 的模型列表才包含。
            with urllib.request.urlopen(_ollama_host() + "/api/tags", timeout=10) as r:
                tags = json.loads(r.read().decode("utf-8"))
            for item in tags.get("models", []):
                if item.get("name") == LLM_MODEL or item.get("model") == LLM_MODEL:
                    digest = item.get("digest") or ""
                    out["model_digest_source"] = "api/tags"
                    break
        out["model_digest"] = digest[:16] or None
        out["model_quant"] = det.get("quantization_level")
        out["model_params"] = det.get("parameter_size")
    except Exception as e:
        out["model_digest"] = "probe-failed: %s" % type(e).__name__
    _ENV_FP_CACHE.update(out)
    return out


def runtime_fingerprint():
    """运行时配置指纹：读的是模块全局量，因此 run_eval_batch.py 用
       --set-gate / --set-dynamic 注入后的\"实际生效值\"会被如实记录。"""
    return {
        "code_sha": _code_sha(),
        "chunk_sha": chunking_fingerprint(),
        "llm_model": LLM_MODEL, "embed_model": EMBED_MODEL,
        "top_k": TOP_K,
        "context_budget": CONTEXT_BUDGET, "budget_escalated": BUDGET_ESCALATED,
        "dynamic_budget": DYNAMIC_BUDGET,
        "escalate_sim_gate": ESCALATE_SIM_GATE,
        "gate_by_subject": GATE_BY_SUBJECT,
        "gate_effective": resolve_gate(),
        "gate_subject": (read_manifest() or {}).get("subject"),
        "relevance_trim": RELEVANCE_TRIM,
        "num_predict": NUM_PREDICT, "temperature": TEMPERATURE,
        "generate_calls": GEN_STATS["total"],
        "generate_fallback_calls": GEN_STATS["fallback"],
        "generate_first_error": GEN_STATS["first_error"],
        "vl_prose": VL_PROSE,
        "vl_quota": VL_QUOTA,
        "query_expand": QUERY_EXPAND,
        "prompt_variant": PROMPT_VARIANT,
        "prompt_sha": _sha(PROMPT),
        "env": env_fingerprint(),
    }


def _subject_of(path):
    """从 books/<学科>/<书名> 的父目录名取学科。取不到返回 None，
       此时 resolve_gate() 回退到全局保守值。"""
    try:
        # 同时兼容 / 与 \ ——不要用 os.path.dirname，它在非 Windows 上不切反斜杠，
        # 会让跨平台测试静默失效
        parts = [x for x in re.split(r"[\\/]+", str(path)) if x]
        d = parts[-2] if len(parts) >= 2 else ""
        # data/input 等只是容器目录，不能被误写成学科；真正的学科目录仍原样保留，
        # 未在分档表中的新学科会由 resolve_gate() 安全回退到全局闸门。
        generic = {"books", "data", "input", "inputs", "document", "documents",
                   "doc", "docs", "file", "files", "source", "sources", "."}
        return d if d and d.casefold() not in generic else None
    except Exception:
        return None


def _manifest_path():
    return os.path.join(DB_PATH, MANIFEST_NAME)


def read_manifest():
    try:
        with open(_manifest_path(), "r", encoding="utf-8") as f:
            m = json.load(f)
        return m if isinstance(m, dict) else None
    except Exception:
        return None


def write_manifest(reset=False, **part):
    """建库时落盘。reset=True 用于 build()（替换建库），False 用于 append 类建库。"""
    m = {} if reset else (read_manifest() or {})
    m["chunk_sha"] = chunking_fingerprint()
    m["code_sha"] = _code_sha()
    m["heading_re"] = HEADING_RE.pattern
    m["chunk_target"], m["chunk_max"] = CHUNK_TARGET, CHUNK_MAX
    m["embed_model"] = EMBED_MODEL
    m["built_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    m.setdefault("parts", []).append(part)
    if part.get("subject"):
        m["subject"] = part["subject"]      # 提到顶层，resolve_gate() 直接读
    m["n_chunks_total"] = sum(int(p.get("n_chunks", 0)) for p in m["parts"])
    try:
        os.makedirs(DB_PATH, exist_ok=True)
        with open(_manifest_path(), "w", encoding="utf-8") as f:
            json.dump(m, f, ensure_ascii=False, indent=1)
    except Exception as e:
        sys.stderr.write("[警告] 建库清单写入失败（不影响建库）：%s\n" % e)
    return m


def _stamp_collection(col, extra=None):
    """把指纹同时写进 chroma 集合元数据——万一 sidecar 丢了还能查。"""
    try:
        meta = {"chunk_sha": chunking_fingerprint(), "code_sha": _code_sha(),
                "built_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        if extra:
            meta.update({k: str(v) for k, v in extra.items()})
        col.modify(metadata=meta)
    except Exception:
        pass   # 老版本 chroma 不支持 modify(metadata=)，静默跳过，sidecar 仍在


def check_library(verbose=True):
    """核对\"当前代码的分块逻辑\"与\"建库时的分块逻辑\"是否一致。
       返回 (ok, info)。ok=False 表示库是别的代码建的，
       此时任何评测数字都不能归因给当前代码。"""
    cur = chunking_fingerprint()
    m = read_manifest()
    if not m:
        ok, status, lib_sha = False, "no_manifest", None
    else:
        lib_sha = m.get("chunk_sha")
        ok = (lib_sha == cur)
        status = "ok" if ok else "stale_library"
    info = {"library_ok": ok, "status": status,
            "runtime_chunk_sha": cur, "library_chunk_sha": lib_sha,
            "library_built_at": (m or {}).get("built_at"),
            "library_code_sha": (m or {}).get("code_sha"),
            "library_n_chunks": (m or {}).get("n_chunks_total"),
            "library_parts": (m or {}).get("parts")}
    if not ok and verbose and QUIET_LIB:
        sys.stderr.write("%s %s runtime=%s library=%s\n" % (STALE_MARK, status, cur, lib_sha))
    elif not ok and verbose:
        bar = "=" * 68
        body = (
            "[库指纹不一致] {st}\n"
            "  当前代码分块指纹 : {cur}\n"
            "  建库时分块指纹   : {lib}\n"
            "  建库时间         : {at}\n"
            "  建库代码 sha     : {csha}\n"
            "  库内总块数       : {nch}\n"
            "  含义：这个向量库不是当前这份代码建的。分块改动存在库里而非代码里，\n"
            "        直接跑评测会得到上一版代码的结果，且产物里看不出来。\n"
            "  处理：重新 build 一遍再评测；或确认这就是你要的对照组。\n"
            "  设 DISTILL_STRICT_LIB=1 可让此情况直接中止。\n"
        ).format(st=status, cur=cur, lib=lib_sha,
                 at=info["library_built_at"], csha=info["library_code_sha"],
                 nch=info["library_n_chunks"])
        sys.stderr.write("\n" + bar + "\n" + body + bar + "\n\n")
    if not ok and STRICT_LIB:
        sys.exit("[中止] DISTILL_STRICT_LIB=1 且库指纹不一致（%s）" % status)
    return ok, info


def library_fingerprint():
    """给评测脚本用的一站式取数：把它整个塞进 _summary.json 即可自证。
       用法（run_eval_batch.py 里加两行）：
           import main
           summary_obj["_fingerprint"] = main.library_fingerprint()
       注意：应在\"跑完所有题之后\"调用，这样 generate_fallback_calls 才是全程累计值。"""
    ok, info = check_library(verbose=False)
    info["runtime"] = runtime_fingerprint()
    return info


def print_fingerprint():
    ok, info = check_library(verbose=False)
    rt = runtime_fingerprint()
    print("== 库指纹 ==")
    print("  向量库路径     : %s" % os.path.abspath(DB_PATH))
    print("  状态           : %s" % ("一致 OK" if ok else "不一致 <-- %s" % info["status"]))
    print("  运行时分块指纹 : %s" % info["runtime_chunk_sha"])
    print("  建库时分块指纹 : %s" % info["library_chunk_sha"])
    print("  建库时间       : %s" % info["library_built_at"])
    print("  建库代码 sha   : %s" % info["library_code_sha"])
    print("  库内总块数     : %s   <-- 与历史对照：Think Python 原版 1441 / 修复① 1118 / 修复② 1135"
          % info["library_n_chunks"])
    for p in (info["library_parts"] or []):
        print("    · %s" % json.dumps(p, ensure_ascii=False))
    print("\n== 运行时配置 ==")
    for k in ("code_sha", "llm_model", "embed_model", "top_k", "context_budget",
              "budget_escalated", "dynamic_budget", "escalate_sim_gate",
              "relevance_trim", "gate_by_subject", "gate_subject", "gate_effective",
              "num_predict", "temperature",
              "vl_prose", "vl_quota", "query_expand", "prompt_variant", "prompt_sha"):
        print("  %-18s: %s" % (k, rt[k]))
    print("\n== 外部依赖（v6 的根因就藏在这里）==")
    for k, v in rt["env"].items():
        print("  %-18s: %s" % (k, v))
    print("\n== 生成调用 ==")
    print("  总调用 / 降级到 /api/chat : %d / %d" % (rt["generate_calls"], rt["generate_fallback_calls"]))
    print("  首次降级异常              : %s" % rt["generate_first_error"])
    if not ok:
        print("\n[结论] 库与当前代码不匹配，先重新 build 再评测。")


def build(pdf, max_pages, vl_limit, use_vl, vl_from=1):
    if not os.path.exists(pdf):
        sys.exit("[错误] 找不到 PDF：%s" % pdf)

    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION)

    cache = load_vl_cache()
    cache_key = lambda p: "%s::p%d" % (os.path.basename(pdf), p)
    _pref = os.path.basename(pdf) + "::"
    _resumed = sum(1 for kk in cache if kk.startswith(_pref))
    if _resumed:
        print("== 断点续传：检测到已缓存 %d 页 VL 描述，将跳过重复解析 ==" % _resumed)

    d = fitz.open(pdf)
    n = min(len(d), max_pages)
    text_chunks, vl_chunks, vl_done = [], [], 0
    t0 = time.time()
    _vr = ("VL上限%d" % vl_limit) if vl_from <= 1 else ("VL上限%d，自第%d页起挑含图页" % (vl_limit, vl_from))
    print("== 混合路由入库：%s（前%d页，%s，%s）==" %
          (os.path.basename(pdf), n, _vr, "纯文本" if not use_vl else "文本+VL"))

    for i in range(n):
        page = d[i]
        txt = page.get_text("text").strip()
        if txt:
            for ch in semantic_chunks("p%d: %s" % (i + 1, txt)):
                text_chunks.append((ch, i + 1, "text"))
        if use_vl and (i + 1) >= vl_from and page.get_images() and vl_done < vl_limit:
            k = cache_key(i + 1)
            if k in cache:
                vtxt = cache[k]
            else:
                print("  VL 第%d页..." % (i + 1), end="", flush=True)
                t1 = time.time()
                vtxt = vl_parse(page.get_pixmap(dpi=VL_DPI).tobytes("png"))
                cache[k] = vtxt
                save_vl_cache(cache)   # 断点续传：每处理完一页立即落盘，中断也不丢
                print(" %.0fs" % (time.time() - t1))
            # 缓存里存 VL 原文，入库前才转写 —— 改转写逻辑无需重新读图
            for ch in semantic_chunks("FIGURE p%d: %s" % (i + 1, _vl_to_prose(vtxt))):
                vl_chunks.append((ch, i + 1, "figure"))
            vl_done += 1
    d.close()
    save_vl_cache(cache)

    alldocs = text_chunks + vl_chunks
    for s in range(0, len(alldocs), 64):
        batch = alldocs[s:s + 64]
        col.add(
            ids=["c%d" % j for j in range(s, s + len(batch))],
            embeddings=embed([b[0] for b in batch]),
            documents=[b[0] for b in batch],
            metadatas=[{"page": b[1], "type": b[2], "source": os.path.basename(pdf)} for b in batch],
        )
        print("  入库 %d/%d" % (min(s + 64, len(alldocs)), len(alldocs)), end="\r")
    print()
    print("完成：文本块 %d | VL块 %d（来自%d个含图页）| 共 %d 块 | 耗时 %.0fs" %
          (len(text_chunks), len(vl_chunks), vl_done, len(alldocs), time.time() - t0))
    write_manifest(reset=True, kind="pdf", source=os.path.basename(pdf),
                   subject=_subject_of(pdf),
                   n_chunks=len(alldocs), n_text=len(text_chunks), n_vl=len(vl_chunks),
                   max_pages=n, use_vl=bool(use_vl), vl_limit=vl_limit, vl_from=vl_from)
    _stamp_collection(col, {"source": os.path.basename(pdf), "n_chunks": len(alldocs)})
    print("向量库已存到 %s" % DB_PATH)
    print("分块指纹 %s | 总块数 %d  <-- 评测前请与建库时的这两个数核对"
          % (chunking_fingerprint(), len(alldocs)))


# ----------------------------- 音频建库 -----------------------------
def build_audio(audio, max_seconds=None, asr_model=None):
    """音频转写入库：append 到现有 knowledge_base（可与 PDF 库共存成混合库）。"""
    import asr  # 延迟导入，避免 PDF 用户触发 faster_whisper
    if not os.path.exists(audio):
        sys.exit("[错误] 找不到音频：%s" % audio)

    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_or_create_collection(COLLECTION)   # 不删库，append

    t0 = time.time()
    print("== 音频转写入库：%s（faster-whisper-small, CPU/int8%s）==" %
          (os.path.basename(audio), ("，前%ds" % max_seconds) if max_seconds else ""))
    docs, info = asr.transcribe_docs(audio, model_dir=asr_model, max_seconds=max_seconds)
    print("  语言 %s | 时长 %.0fs | 生成音频块 %d 个" % (info.language, info.duration, len(docs)))

    base = os.path.basename(audio)
    safe = re.sub(r"[^0-9A-Za-z_.-]", "_", base)
    for s in range(0, len(docs), 64):
        batch = docs[s:s + 64]
        col.upsert(   # upsert：同名音频重跑不报错、直接覆盖
            ids=["audio_%s_%d" % (safe, j) for j in range(s, s + len(batch))],
            embeddings=embed([b[0] for b in batch]),
            documents=[b[0] for b in batch],
            metadatas=[{"page": b[1], "type": b[2], "source": base, "time": b[3]} for b in batch],
        )
        print("  入库 %d/%d" % (min(s + 64, len(docs)), len(docs)), end="\r")
    print()
    print("完成：音频块 %d（append 到 %s）| 耗时 %.0fs" % (len(docs), COLLECTION, time.time() - t0))
    write_manifest(kind="audio", source=os.path.basename(audio),
                   subject=_subject_of(audio), n_chunks=len(docs))
    _stamp_collection(col)
    print("向量库已存到 %s" % DB_PATH)


# ----------------------------- EPUB 建库 -----------------------------
def _epub_blocks(path, min_chars=40):
    """EPUB -> [(text, chapter_idx, loc)]。按 spine 顺序遍历，每章一块。
       loc 沿用引用锚定口径：ch{序号}:{章节标题}（EPUB 无固定页码）。"""
    from ebooklib import epub, ITEM_DOCUMENT
    from bs4 import BeautifulSoup

    book = epub.read_epub(path)
    toc_title = {}
    def _walk(items):
        for it in items:
            if isinstance(it, tuple):
                link, children = it[0], it[1]
                if getattr(link, "href", None):
                    toc_title[link.href.split("#")[0]] = link.title
                _walk(children)
            elif getattr(it, "href", None):
                toc_title[it.href.split("#")[0]] = it.title
    try:
        _walk(book.toc)
    except Exception:
        pass

    out, idx = [], 0
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        fname = item.get_name()
        if "nav" in fname.lower():
            continue
        soup = BeautifulSoup(item.get_content(), "html.parser")
        head = soup.find(["h1", "h2"])
        if head:
            title = head.get_text(strip=True)
        elif toc_title.get(fname):
            title = toc_title.get(fname)
        else:
            title = os.path.splitext(os.path.basename(fname))[0]
        for t in soup(["script", "style"]):
            t.decompose()
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
        if len(text) < min_chars:
            continue
        idx += 1
        out.append((text, idx, "ch%d:%s" % (idx, title or "?")))
    return out


def build_epub(epub_path, max_chapters=None):
    """EPUB 转文本入库：append 到现有 knowledge_base（与 PDF/音频共存成混合库）。"""
    if not os.path.exists(epub_path):
        sys.exit("[错误] 找不到 EPUB：%s" % epub_path)

    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_or_create_collection(COLLECTION)   # 不删库，append

    t0 = time.time()
    print("== EPUB 转文本入库：%s ==" % os.path.basename(epub_path))
    blocks = _epub_blocks(epub_path)
    if max_chapters:
        blocks = blocks[:max_chapters]
    # 每章再过语义分块，保持与 PDF 一致的块粒度
    docs = []
    for text, ch_idx, loc in blocks:
        for chk in semantic_chunks(text):
            docs.append((chk, ch_idx, loc))
    print("  章节 %d | 生成文本块 %d" % (len(blocks), len(docs)))

    base = os.path.basename(epub_path)
    safe = re.sub(r"[^0-9A-Za-z_.-]", "_", base)
    for s in range(0, len(docs), 64):
        batch = docs[s:s + 64]
        col.upsert(
            ids=["epub_%s_%d" % (safe, j) for j in range(s, s + len(batch))],
            embeddings=embed([b[0] for b in batch]),
            documents=[b[0] for b in batch],
            metadatas=[{"page": b[1], "type": "epub", "source": base, "loc": b[2]} for b in batch],
        )
        print("  入库 %d/%d" % (min(s + 64, len(docs)), len(docs)), end="\r")
    print()
    print("完成：EPUB 块 %d（append 到 %s）| 耗时 %.0fs" % (len(docs), COLLECTION, time.time() - t0))
    write_manifest(kind="epub", source=os.path.basename(epub_path),
                   subject=_subject_of(epub_path), n_chunks=len(docs))
    _stamp_collection(col)
    print("向量库已存到 %s" % DB_PATH)


# ----------------------------- 独立图片建库 -----------------------------
def build_image(img_path):
    """独立图片入库：走 VL 通道生成描述，append 到现有 knowledge_base。"""
    if not os.path.exists(img_path):
        sys.exit("[错误] 找不到图片：%s" % img_path)
    ext = os.path.splitext(img_path)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg", ".svg"):
        sys.exit("[错误] 不支持的图片格式：%s（仅 PNG/JPG/SVG）" % ext)

    client = chromadb.PersistentClient(path=DB_PATH)
    col = client.get_or_create_collection(COLLECTION)   # 不删库，append

    t0 = time.time()
    base = os.path.basename(img_path)
    print("== 独立图片入库：%s（走 VL 解析）==" % base)
    if ext == ".svg":
        # Ollama 的 images 字段不直接接收 SVG；使用已有 PyMuPDF 在本地栅格化，
        # 保留矢量文字/公式结构后再送入 VL，不引入 Cairo 等系统依赖。
        svg_doc = None
        try:
            svg_doc = fitz.open(img_path)
            if len(svg_doc) < 1:
                raise ValueError("SVG 没有可渲染页面")
            image_bytes = svg_doc[0].get_pixmap(dpi=VL_DPI, alpha=False).tobytes("png")
        except Exception as e:
            sys.exit("[错误] SVG 栅格化失败：%s" % e)
        finally:
            if svg_doc is not None:
                svg_doc.close()
    else:
        image_bytes = open(img_path, "rb").read()
    b64 = base64.b64encode(image_bytes).decode()
    p = ("Describe this image in detail for a knowledge base: list every text label, "
         "component, or figure element, then briefly explain what the image shows.")
    desc = ""
    for attempt in range(VL_RETRY):
        try:
            desc = _strip_think(_chat_vl(VL_MODEL, p, b64))
            break
        except Exception:
            print(" [VL重试%d]" % (attempt + 1), end="", flush=True); time.sleep(3)
    if not desc or desc == "[NO REFERENCE FOUND]":
        sys.exit("[错误] VL 解析图片失败（模型未返回描述）")

    loc = "image:%s" % base
    docs = [(chk, 0, loc) for chk in semantic_chunks(desc)] or [(desc, 0, loc)]
    safe = re.sub(r"[^0-9A-Za-z_.-]", "_", base)
    col.upsert(
        ids=["image_%s_%d" % (safe, j) for j in range(len(docs))],
        embeddings=embed([b[0] for b in docs]),
        documents=[b[0] for b in docs],
        metadatas=[{"page": 0, "type": "image", "source": base, "loc": loc} for _ in docs],
    )
    print("完成：图片块 %d（append 到 %s）| 耗时 %.0fs" % (len(docs), COLLECTION, time.time() - t0))
    write_manifest(kind="image", source=os.path.basename(img_path),
                   subject=_subject_of(img_path), n_chunks=len(docs))
    _stamp_collection(col)
    print("向量库已存到 %s" % DB_PATH)


# ----------------------------- 智能体配置生成 -----------------------------
def _infer_subject(docs, book):
    """采样库内容，让 LLM 判断学科 + 生成助教角色；LLM 不可用则按书名降级。"""
    excerpt = "\n".join((d or "")[:300] for d in docs[:8])
    prompt = ('Based on these textbook excerpts, reply with ONLY a JSON object '
              '{"subject":"<2-4 word academic subject>","role":"<one-sentence teaching-assistant role>"}.\n\n'
              "Excerpts:\n" + excerpt)
    try:
        out = _generate(LLM_MODEL, prompt, options={"temperature": 0, "num_predict": 120})
        m = re.search(r"\{.*\}", out["response"], re.S)
        obj = json.loads(m.group(0))
        subj = (obj.get("subject") or "").strip()
        role = (obj.get("role") or "").strip()
        if subj and role:
            return subj, role
    except Exception:
        pass
    base = os.path.splitext(book)[0]
    return base, "a teaching assistant for %s" % base


def _compose_system(subject, role, book):
    return (
        'You are %s. You assist users with the material "%s" (domain: %s).\n'
        "Rules:\n"
        "- Answer ONLY using the retrieved material provided at query time.\n"
        "- Always cite sources by copying the bracket tag shown above each block "
        "exactly as given (e.g. [p.112] for pages, [ch2:Title] for ebook chapters, "
        "[audio mm:ss] for audio). Never invent page numbers.\n"
        '- If the retrieved material gives no basis, answer exactly "[NO REFERENCE FOUND]".\n'
        "- For arithmetic, a calculator tool computes the exact result; report it faithfully.\n"
        "- Be precise, concise, and educational."
    ) % (role, book, subject)


def gen_agent(pdf=None, max_pages=120, vl_limit=15, use_vl=True):
    """生成标准化智能体包：system prompt + 工具链(检索/计算/引用) + Ollama 运行配置。"""
    if pdf:
        build(pdf, max_pages, vl_limit, use_vl)      # 给了 PDF 就先建库
    col = get_collection()

    sample = col.get(limit=12)
    docs = (sample.get("documents") or [])[:12]
    metas = sample.get("metadatas") or []
    sources = sorted({m.get("source", "?") for m in metas}) or ["knowledge_base"]
    book = sources[0]
    print("== 生成智能体配置：采样 %d 块，判定学科中... ==" % len(docs))
    subject, role = _infer_subject(docs, book)
    system_text = _compose_system(subject, role, book)
    print("   学科：%s" % subject)

    safe = re.sub(r"[^0-9A-Za-z_.-]", "_", os.path.splitext(book)[0]) or "agent"
    outdir = os.path.abspath("./agent_%s" % safe)
    os.makedirs(outdir, exist_ok=True)
    code_dir = os.path.dirname(os.path.abspath(__file__))
    db_abs = os.path.abspath(DB_PATH)
    model_name = ("%s-assistant" % safe).lower()

    open(os.path.join(outdir, "system_prompt.txt"), "w", encoding="utf-8").write(system_text)

    modelfile = ('FROM %s\n\nSYSTEM """%s"""\n\n'
                 "PARAMETER temperature %s\nPARAMETER num_predict %s\n"
                 % (LLM_MODEL, system_text, TEMPERATURE, NUM_PREDICT))
    open(os.path.join(outdir, "Modelfile"), "w", encoding="utf-8").write(modelfile)

    runbat = (
        "@echo off\r\n"
        "chcp 65001 >nul\r\n"
        "echo == 1) 创建专属 Ollama 模型（system prompt + 参数）==\r\n"
        'ollama create %s -f "%s\\Modelfile"\r\n'
        "echo == 2) 启动智能体对话（工具：检索 + 计算器）==\r\n"
        'python "%s\\agent_runtime.py" --db "%s" --collection %s --prompt "%s\\system_prompt.txt" --model %s\r\n'
        % (model_name, outdir, code_dir, db_abs, COLLECTION, outdir, model_name)
    )
    open(os.path.join(outdir, "run.bat"), "w", encoding="utf-8", newline="").write(runbat)

    readme = (
        "# 智能体包：%s\n\n"
        "- 学科：%s\n"
        "- 工具链：检索（RAG + 引用溯源）、计算器（白名单 AST 安全求值）；规则路由\n"
        "- 一键启动：双击 `run.bat`（先 `ollama create` 专属模型，再进入对话）\n"
        "- 手动启动对话：\n"
        '  `python "%s\\agent_runtime.py" --db "%s" --prompt system_prompt.txt --model %s`\n\n'
        "对话中：普通问题走检索问答（带 [p.X] / [audio mm:ss] 引用）；"
        "输入算式（如 `3*(4+5)`、`sqrt(144)`）走计算器，返回精确结果。\n"
        % (model_name, subject, code_dir, db_abs, model_name)
    )
    open(os.path.join(outdir, "README.md"), "w", encoding="utf-8").write(readme)

    print("完成：智能体包已生成到 %s" % outdir)
    print("  ├─ system_prompt.txt   （按书定制 · 含引用约束）")
    print("  ├─ Modelfile           （Ollama 运行配置）")
    print("  ├─ agent_runtime.py    （运行时在 code/，检索+计算器+路由）")
    print("  ├─ run.bat             （一键：create 模型 + 启动对话）")
    print("  └─ README.md")
    print("启动：双击 %s\\run.bat" % outdir)


# ----------------------------- 提问 -----------------------------
def get_collection():
    if not os.path.exists(DB_PATH):
        sys.exit("[错误] 还没建库，请先运行：python main.py build --pdf 你的书.pdf")
    client = chromadb.PersistentClient(path=DB_PATH)
    try:
        col = client.get_collection(COLLECTION)
    except Exception:
        sys.exit("[错误] 向量库为空，请先 build。")
    check_library(verbose=True)   # 库是旧代码建的就大声告警（STRICT 模式下直接中止）
    return col


_ABSTAIN_RE = re.compile(r"no\s+(?:\w+\s+){0,2}references?\s+(?:found|available|provided|in)", re.I)


def is_abstain(answer):
    # 空/纯空白也视为拒答（模型只输出 think 块时 content 为空）
    if answer is None or not str(answer).strip():
        return True
    """鲁棒判定拒答：[NO REFERENCE FOUND] 及常见变体。"""
    a = answer.lower()
    return ("no reference found" in a
            or bool(re.search(r"\[[^\]]*\bno\b[^\]]*\breferences?\b", a))
            or bool(_ABSTAIN_RE.search(a)))


def _cosine(a, b):
    s = aa = bb = 0.0
    for x, y in zip(a, b):
        s += x * y; aa += x * x; bb += y * y
    if aa == 0 or bb == 0:
        return 0.0
    return s / (aa ** 0.5 * bb ** 0.5)


def _pack_relevance(docs, question, budget):
    """动态裁剪：沿检索相关度顺序整块贪心保留，绝不切半块。
       docs 已由 col.query 按相关度降序返回，直接信任该顺序（不再二次重排——
       二次余弦重排会与检索排序打架，在仅容一块时反而选错块）。
       相对字符截断的两点改进：①整块保留不砍半 ②跳过塞不下的大块、纳入后面更小的相关块。
       question 参数保留仅为兼容调用签名。
       返回 (拼好的块列表, 原始下标列表)——下标供引用溯源。"""
    if not docs:
        return [], []
    # 若连检索序最相关的第0块都整块塞不下 -> 截断第0块（与纯截断同行为），
    # 不跳过它去纳入后面更小的块（否则会捡了噪声小块、丢了最相关块=答案）。
    if len(docs[0]) > budget:
        return [docs[0][:budget]], [0]
    picked, used, sep = [], 0, len("\n---\n")
    for i in range(len(docs)):                # docs 已按相关度降序
        need = len(docs[i]) + (sep if picked else 0)
        if used + need <= budget:
            picked.append(i); used += need
    return [docs[i] for i in picked], picked  # picked 已升序（检索序）


def _pack_truncate(docs, budget):
    """旧策略：按检索序拼接，超预算按字符截断（保留用于对照评测）。
       返回 (块列表, 下标列表)。"""
    packed, idx, used = [], [], 0
    for i, doc in enumerate(docs):
        if used + len(doc) > budget:
            if budget - used > 120:
                packed.append(doc[:budget - used]); idx.append(i)
            break
        packed.append(doc); idx.append(i); used += len(doc)
    return packed, idx



# ============ 引用准确度：真页标签 + 引用校验（检验功能）============
def _cite_tag(m):
    """一个检索块的\"引用标签\"：喂进 context 让模型照标，也用于事后校验。
       页码型→ p.112；EPUB/图片→ 用 loc（ch2:Viruses / image:fig.png）；音频→ audio mm:ss。"""
    t = m.get("type")
    if t == "audio":
        return "audio %s" % m.get("time", "?")
    if t in ("epub", "image"):
        return m.get("loc", t)
    return "p.%d" % m["page"]


def _labeled_context(packed, packed_idx, metas=None):
    """给每块前缀真实来源标签，让模型引用真实页码而非瞎编。
       metas=None（评测脚本）时退回纯拼接，行为与旧版一致。"""
    if not metas:
        return "\n---\n".join(packed)
    blocks = []
    for pos, i in enumerate(packed_idx):
        tag = _cite_tag(metas[i]) if i < len(metas) else "?"
        blocks.append("[%s]\n%s" % (tag, packed[pos]))
    return "\n---\n".join(blocks)


def _format_prompt(context, question, packed_idx=None, metas=None):
    """统一格式化问答 prompt，供 CLI、Web UI 和生成智能体共同使用。

    ``PROMPT`` 的所有变体都要求 ``tag_example``；集中在这里生成可避免
    调用方漏传占位符，也确保 PDF/EPUB/音频/图片展示各自真实的引用格式。
    """
    tags = []
    if metas:
        for i in (packed_idx or []):
            if i < len(metas):
                tags.append(_cite_tag(metas[i]))
    uniq = list(dict.fromkeys(tags))[:2]
    tag_example = " or ".join("[%s]" % t for t in uniq) if uniq else "[p.112]"
    return PROMPT.format(context=context, question=question, tag_example=tag_example)


def _norm_cite(s):
    """归一化一个引用/标签：'p.112'/'p112'/'P.112(text)' → 'p.112'；其余（ch../image../audio..）小写去空格。"""
    s = s.strip().lower().replace(" ", "").split("(")[0]
    mm = re.match(r"p\.?(\d+)", s)
    return ("p.%s" % mm.group(1)) if mm else s


def verify_citations(answer, packed_idx, metas):
    """检验功能：核对正文里的 [p.X] 等引用，是否都落在\"实际喂给模型的检索块\"里。
       引用了检索块里没有的页 = 疑似编造，标记出来。返回校验详情 + 引用准确率。"""
    metas = metas or []
    valid = {_norm_cite(_cite_tag(metas[i])) for i in packed_idx if i < len(metas)}
    cited = []
    for raw in re.findall(r"\[([^\]]+)\]", str(answer)):
        n = _norm_cite(raw)
        if re.match(r"(p\.\d+|ch|image|audio)", n):   # 只算引用型括号，忽略 [NO REFERENCE FOUND] 等
            cited.append(n)
    cset = list(dict.fromkeys(cited))                   # 去重保序
    hit = [c for c in cset if c in valid]
    bad = [c for c in cset if c not in valid]
    total = len(cset)
    abstained = is_abstain(answer)
    missing = total == 0 and not abstained
    return {
        "total": total,
        "hit": hit,
        "fabricated": bad,
        "valid_sources": sorted(valid),
        "missing": missing,
        "ok": (not missing) and (not bad),
        "rate": (len(hit) / total) if total else (1.0 if abstained else 0.0),
    }


def _cite_check_line(cc):
    if cc["total"] == 0:
        return "正文无显式引用  <-- 警告" if cc.get("missing") else "拒答，无需引用"
    if cc["ok"]:
        return "%d/%d 引用全部命中检索来源 OK" % (len(cc["hit"]), cc["total"])
    return "%d/%d 命中；疑似编造: %s  <-- 警告" % (len(cc["hit"]), cc["total"], "、".join(cc["fabricated"]))


def _run_once(docs, question, budget, metas=None):
    """按给定预算打包检索块并生成一次。
       metas 提供时给每块前缀真实来源标签（让模型引用真页码，杜绝瞎编）；不提供则纯拼接（兼容评测脚本）。
       返回 (答案, tokens, 打包块的原始下标列表)——下标供引用溯源精确对齐。"""
    if RELEVANCE_TRIM:
        packed, packed_idx = _pack_relevance(docs, question, budget)
    else:
        packed, packed_idx = _pack_truncate(docs, budget)
    context = _labeled_context(packed, packed_idx, metas)
    out = _generate(LLM_MODEL,
                    _format_prompt(context, question, packed_idx, metas),
                    options={"temperature": TEMPERATURE, "num_predict": NUM_PREDICT})
    toks = out.get("prompt_eval_count", 0) + out.get("eval_count", 0)
    return out["response"].strip(), toks, packed_idx


# 分型检索配额：给 VL/图片块保留的席位数。0 = 关闭（v7full 口径）。
# 由 DISTILL_VL_QUOTA 注入，便于 A/B。
# 动机：混合库里 VL 块占比极低（A&P 实测 122/9285 = 1.3%），且图内标签是碎片、
#       检索距离天然差于完整文本段落，按距离排序永远进不了 Top5。
#       45 道图题实测：VL 块已入库，但 39/45 没被检索命中。
# 注意：这是查询期参数，不影响入库，因此不计入 chunking_fingerprint()。
VL_QUOTA = int(os.environ.get("DISTILL_VL_QUOTA", "0") or 0)

# 短查询扩写：查询词数 <= 此值时启用；0 = 关闭（v7full 口径）。由 DISTILL_QUERY_EXPAND 注入。
# 动机（v7full 实测）：按查询词数分档的严格命中率——
#   1-3 词 61.3%（972 题） / 4-6 词 92.1% / 7-10 词 92.2% / 11+ 词 74.7%
# 短查询是全系统最大的缺口（31pp）。原因是 2-3 个词的向量太稀疏，检索取不准。
# 做法：先用原查询检索，取 top-1 块的开头补进查询做第二次检索，两次结果按距离归并。
# 不调 LLM——加一次 embedding（约 30ms）远比加一次生成（约 2s）划算。
# 已知风险：若 top-1 本身取错，扩写会强化该错误（HyDE 的经典失效模式），
#          因此保留原查询结果、只做归并不做替换，且默认关闭、需 A/B 验证后再开。
QUERY_EXPAND = int(os.environ.get("DISTILL_QUERY_EXPAND", "0") or 0)


def _q_wordcount(q):
    return len(re.findall(r"[A-Za-z0-9']+", q or ""))


def _expand_query(question, docs):
    """用 top-1 块的前两句把短查询补长。返回 None 表示无法扩写。"""
    if not docs:
        return None
    body = re.sub(r"^\s*(\[[^\]]*\]|p\d+:)\s*", "", docs[0]).strip()
    sents = split_sentences(body)[:2]
    if not sents:
        return None
    return ("%s %s" % (question, " ".join(sents)))[:400]


def _retrieve(col, qv, question=None):
    """检索。QUERY_EXPAND>0 且查询过短时做一次扩写重检索并按距离归并；
       VL_QUOTA>0 时额外单独召回图块，并强制插到次席。

    为什么要强制插位而不是"多召回几个"：多召回的块若排在末尾，
    _pack_relevance 在 900 字符预算下（约容 2 块）根本轮不到它们，
    等于没召回。插到第 1 位之后，既保留检索序最优的那块，又保证图块能进上下文。
    代价是挤掉一个文本块的席位——所以默认关闭，由 A/B 数据决定是否开启。
    """
    res = col.query(query_embeddings=[qv], n_results=TOP_K)
    docs, metas = res["documents"][0], res["metadatas"][0]
    dists = (res.get("distances") or [[]])[0]

    # ---- 短查询扩写 ----
    # 归并必须重排而不是简单追加：_pack_relevance 直接信任检索序、
    # 900 预算下只装得下约两块，追加到末尾的候选根本轮不到（VL_QUOTA 已踩过这个坑）。
    #
    # 【2026-07-31 修】第一版按距离排序归并，实测净 -5 道、过度拒答 5->12。
    # 根因：两次检索的距离来自不同查询向量，**不可比**。扩写后的查询更长更具体，
    # 对自己邻居的距离系统性更低，于是整体挤掉原查询的结果——
    # `cornea blindness condition` 那道，正确块 p955 本在原查询 top-8 内（v7full 正是靠它答对），
    # 被扩写结果挤出归并后的 top-8。
    # 改用 RRF（倒数排名融合）：按**名次**而非距离融合，score = Σ 1/(k + rank)。
    # 名次是序数，跨查询可比，这正是多路检索融合的标准解法。
    # 另加两道保险：原查询 top-1 强制保留；扩写结果只能补位不能挤掉原 top-3。
    if (QUERY_EXPAND > 0 and question and docs and dists
            and _q_wordcount(question) <= QUERY_EXPAND):
        exp = _expand_query(question, docs)
        if exp:
            try:
                r2 = col.query(query_embeddings=[embed([exp])[0]], n_results=TOP_K)
                d2, m2 = r2["documents"][0], r2["metadatas"][0]
                s2 = (r2.get("distances") or [[]])[0]
                K = 60                                    # RRF 平滑常数，业界惯用 60
                score, info = {}, {}
                for rank, (dd, mm, ss) in enumerate(zip(docs, metas, dists)):
                    score[dd] = score.get(dd, 0.0) + 1.0 / (K + rank + 1)
                    info[dd] = (mm, ss)
                for rank, (dd, mm, ss) in enumerate(zip(d2, m2, s2)):
                    score[dd] = score.get(dd, 0.0) + 1.0 / (K + rank + 1)
                    info.setdefault(dd, (mm, ss))         # 原查询的距离优先保留
                keep = docs[:3]                           # 原 top-3 锁位，扩写只能补位
                rest = sorted([d for d in score if d not in keep],
                              key=lambda d: -score[d])
                merged = (keep + rest)[:TOP_K]
                docs = merged
                metas = [info[d][0] for d in merged]
                dists = [info[d][1] for d in merged]
            except Exception:
                pass          # 扩写失败就用原结果，不影响主流程

    if VL_QUOTA <= 0 or not docs:
        return docs, metas, dists

    have = {d for d in docs}
    try:
        r2 = col.query(query_embeddings=[qv], n_results=VL_QUOTA,
                       where={"type": {"$in": ["figure", "image"]}})
        vdocs, vmetas = r2["documents"][0], r2["metadatas"][0]
        vdists = (r2.get("distances") or [[]])[0]
    except Exception:
        return docs, metas, dists      # 库里没有图块或旧版 chroma 不支持 where

    ins = 1 if len(docs) > 1 else len(docs)   # 保留检索序最优的那块在首位
    added = 0
    for j, vd in enumerate(vdocs):
        if vd in have:                        # 主检索已经召回了，不重复占位
            continue
        docs.insert(ins + added, vd)
        metas.insert(ins + added, vmetas[j])
        if dists and j < len(vdists):
            dists.insert(ins + added, vdists[j])
        have.add(vd)
        added += 1
    return docs, metas, dists


def should_escalate(answer, docs, dists, dynamic=True):
    """统一判断是否动态升配，避免 CLI、Web UI、智能体使用不同闸门口径。"""
    if not dynamic or not docs or not is_abstain(answer):
        return False
    best_dist = min(dists) if dists else None
    gate = resolve_gate()
    return gate is None or best_dist is None or best_dist <= gate


def ask(col, question, verbose=True, dynamic=DYNAMIC_BUDGET):
    qv = embed([question])[0]
    docs, metas, _d = _retrieve(col, qv, question)
    best_dist = min(_d) if _d else None      # 最优检索块的距离，越小越相关

    # 第一档：预算 900（常态，保 Token 效率）
    answer, toks, packed_idx = _run_once(docs, question, CONTEXT_BUDGET, metas)
    escalated = False
    # 动态升配：仅当"检索有命中却拒答"且通过闸门时，升到 1800 重答一次
    if should_escalate(answer, docs, _d, dynamic):
        ans2, toks2, packed_idx2 = _run_once(docs, question, BUDGET_ESCALATED, metas)
        toks += toks2                      # 两次调用 token 累计，成本如实计
        escalated = True
        if not is_abstain(ans2):           # 升配后答出来了才采用
            answer, packed_idx = ans2, packed_idx2

    if verbose:
        def _src(m):
            t = m.get("type")
            if t == "audio":
                return "audio %s" % m.get("time", "?")
            if t in ("epub", "image"):          # 无页码，用 loc 定位（ch2:Viruses / image:fig.png）
                return m.get("loc", t)
            return "p%d(%s)" % (m["page"], m["type"])
        # 按实际打包的块下标取来源（相关度裁剪下打包的不一定是前几块）
        srcs = sorted({_src(metas[i]) for i in packed_idx if i < len(metas)})
        tag = "  (动态升配 %d)" % BUDGET_ESCALATED if escalated else ""
        if best_dist is not None:
            tag += "  dist=%.4f" % best_dist
        cc = verify_citations(answer, packed_idx, metas)
        print("\n" + answer)
        print("\n[来源] %s  | tokens: %d%s" % ("、".join(srcs), toks, tag))
        print("[引用校验] %s" % _cite_check_line(cc))
    return answer


# ============ brief 文档生成模式（面向业务场景：综合资料生成结构化摘要）============
BRIEF_PROMPT = """You are writing a concise briefing document for a business/professional reader.
Using ONLY the material below, write a structured brief on the given TOPIC.
Requirements:
- Organize as: 【概要】one-sentence summary; 【要点】3-6 bullet points; 【依据】key supporting facts.
- Each block starts with a source tag like [p.112]. When you cite a fact, reuse the tag shown above that block (e.g. [p.112]).
- Do NOT invent anything not in the material. If the material is insufficient, say so plainly.
- Understanding-level accuracy is fine; you need not quote verbatim.

Material:
{context}

Topic: {topic}

Brief:"""


BRIEF_SYSTEM = ("You are a briefing writer. You MUST produce a structured brief based on the given material. "
                "Never output '[NO REFERENCE FOUND]' or refuse; if some detail is missing, write the brief with what IS present.")

def _gen_brief_raw(prompt):
    """brief 专用生成：think=False + 只剥 think 标签、不套 ask 那套\"空则拒答\"兜底，避免长文档被误判拒答。"""
    kw = {"model": LLM_MODEL, "prompt": prompt, "system": BRIEF_SYSTEM,
          "think": False, "options": {"temperature": TEMPERATURE, "num_predict": max(NUM_PREDICT, 700)}}
    try:
        r = ollama.generate(**kw)
    except Exception:
        # 与 _generate 一致的 /api/chat 降级
        payload = {"model": LLM_MODEL, "stream": False, "think": False,
                   "messages": [{"role": "system", "content": BRIEF_SYSTEM},
                                {"role": "user", "content": prompt}],
                   "options": {"temperature": TEMPERATURE, "num_predict": max(NUM_PREDICT, 700)}}
        out = _post_json("/api/chat", payload)
        r = {"response": out.get("message", {}).get("content", ""),
             "prompt_eval_count": out.get("prompt_eval_count", 0), "eval_count": out.get("eval_count", 0)}
    text = re.sub(r"<think>.*?</think>", "", str(r.get("response", "")), flags=re.S | re.I)
    text = re.sub(r"</?think>", "", text, flags=re.I).strip()
    toks = r.get("prompt_eval_count", 0) + r.get("eval_count", 0)
    return text, toks

def _run_once_brief(docs, topic, budget, metas=None):
    """同构复用检索/相关度打包/溯源；生成走 _gen_brief_raw（不套问答拒答兜底）。
       metas 提供时给每块前缀真实来源标签，让 brief 正文引用真页码。"""
    if RELEVANCE_TRIM:
        packed, packed_idx = _pack_relevance(docs, topic, budget)
    else:
        packed, packed_idx = _pack_truncate(docs, budget)
    context = _labeled_context(packed, packed_idx, metas)
    text, toks = _gen_brief_raw(BRIEF_PROMPT.format(context=context, topic=topic))
    return text, toks, packed_idx


def brief(col, topic, verbose=True):
    """面向业务场景：围绕 topic 综合检索资料，生成带出处的结构化 brief 文档。
       复用与 ask 相同的检索/相关度打包/引用溯源；直接用升配预算一步到位。"""
    qv = embed([topic])[0]
    res = col.query(query_embeddings=[qv], n_results=TOP_K)
    docs, metas = res["documents"][0], res["metadatas"][0]
    answer, toks, packed_idx = _run_once_brief(docs, topic, BUDGET_ESCALATED, metas)
    if verbose:
        def _src(m):
            t = m.get("type")
            if t == "audio":
                return "audio %s" % m.get("time", "?")
            if t in ("epub", "image"):
                return m.get("loc", t)
            return "p%d(%s)" % (m["page"], m["type"])
        srcs = sorted({_src(metas[i]) for i in packed_idx if i < len(metas)})
        cc = verify_citations(answer, packed_idx, metas)
        print("\n" + answer)
        print("\n[来源] %s  | tokens: %d" % ("、".join(srcs), toks))
        print("[引用校验] %s" % _cite_check_line(cc))
    return answer
# ============ brief 模式结束 ============


# ----------------------------- 入口 -----------------------------
def main():
    ap = argparse.ArgumentParser(description="知识蒸馏管线 · 混合路由 RAG")
    sub = ap.add_subparsers(dest="cmd")

    b = sub.add_parser("build", help="对一本 PDF 或一段音频做入库")
    b.add_argument("--pdf", help="PDF 路径（混合路由，替换建库）")
    b.add_argument("--max-pages", type=int, default=120)
    b.add_argument("--vl-limit", type=int, default=15)
    b.add_argument("--vl-from", type=int, default=1,
                   help="从第几页起挑含图页给 VL（大部头前几十页多为封面/目录）")
    b.add_argument("--no-vl", action="store_true", help="纯文本模式，跳过 VL")
    b.add_argument("--audio", help="音频路径 MP3/WAV/FLAC（转写后 append 入库）")
    b.add_argument("--max-seconds", type=int, default=None, help="仅取音频前 N 秒（对齐≤5min验收）")
    b.add_argument("--asr-model", default=None, help="faster-whisper 本地模型目录")
    b.add_argument("--epub", help="EPUB 路径（转文本后 append 入库）")
    b.add_argument("--max-chapters", type=int, default=None, help="仅取 EPUB 前 N 章")
    b.add_argument("--image", help="独立图片路径 PNG/JPG/SVG（走 VL 解析后 append 入库）")

    a = sub.add_parser("ask", help="向已建好的库提一个问题")
    a.add_argument("question")

    sub.add_parser("chat", help="连续问答，输入 exit 退出")

    g = sub.add_parser("agent", help="根据已建库/PDF 生成智能体配置包")
    g.add_argument("--pdf", default=None, help="给定则先建库再生成")
    g.add_argument("--max-pages", type=int, default=120)
    g.add_argument("--vl-limit", type=int, default=15)
    g.add_argument("--no-vl", action="store_true", help="纯文本模式，跳过 VL")

    br = sub.add_parser("brief", help="围绕一个主题生成带出处的 brief 文档")
    br.add_argument("topic")

    fp = sub.add_parser("fingerprint", help="打印库指纹 + 运行时配置（评测前自检，不调用模型）")
    fp.add_argument("--json", action="store_true", help="输出 JSON，供评测脚本采集")

    args = ap.parse_args()

    if args.cmd == "build":
        if args.audio:
            build_audio(args.audio, args.max_seconds, args.asr_model)
        elif args.epub:
            build_epub(args.epub, args.max_chapters)
        elif args.image:
            build_image(args.image)
        elif args.pdf:
            build(args.pdf, args.max_pages, args.vl_limit, use_vl=not args.no_vl,
                  vl_from=getattr(args, 'vl_from', 1))
        else:
            sys.exit("[错误] build 需要 --pdf / --audio / --epub / --image")
    elif args.cmd == "ask":
        ask(get_collection(), args.question)
    elif args.cmd == "chat":
        col = get_collection()
        print("进入问答（输入 exit 退出）")
        while True:
            try:
                q = input("\n> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("exit", "quit", ""):
                break
            ask(col, q)
    elif args.cmd == "agent":
        gen_agent(args.pdf, args.max_pages, args.vl_limit, use_vl=not args.no_vl)
    elif args.cmd == "brief":
        brief(get_collection(), args.topic)
    elif args.cmd == "fingerprint":
        if getattr(args, "json", False):
            print(json.dumps(library_fingerprint(), ensure_ascii=False))
        else:
            print_fingerprint()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
