# main.py 两处修复 · 精确改法

> 依据：4432 题全量评测的实测数据
> 改完请先跑单本验证，再全量重跑出 A/B 表

---

## 修复一 · EPUB 引用格式（332 处错误引用 → 预期归零）

### 问题

`_labeled_context()` 已经在给每块打正确标签（EPUB 是 `[ch2:标题]`），**溯源代码没问题**。
问题在 PROMPT **两次用 `[p.112]` 举例**，把模型往页码格式上带。小说的块标着 `[ch2:...]`，模型仍输出 `[p.100]`——而《简·爱》只有 47 章。

### 改动 1／2：PROMPT（第 60 行附近）

**原文：**

```python
PROMPT = """Answer the question using ONLY the material below. Each block starts with a source tag like [p.112]. When you cite, reuse the tag shown above that block (e.g. [p.112]).
If there is no basis in the material, answer exactly "[NO REFERENCE FOUND]".

Material:
{context}

Question: {question}
Answer:"""
```

**改成：**

```python
PROMPT = """Answer the question using ONLY the material below. Each block starts with a source tag in square brackets, for example {tag_example}.
When you cite, copy the tag of the block you actually used EXACTLY as shown above that block. Do NOT invent page numbers and do NOT change the tag format.
If there is no basis in the material, answer exactly "[NO REFERENCE FOUND]".

Material:
{context}

Question: {question}
Answer:"""
```

### 改动 2／2：`_run_once()`（第 668 行附近）

**原文：**

```python
    context = _labeled_context(packed, packed_idx, metas)
    out = _generate(LLM_MODEL, PROMPT.format(context=context, question=question),
                    options={"temperature": TEMPERATURE, "num_predict": NUM_PREDICT})
```

**改成：**

```python
    context = _labeled_context(packed, packed_idx, metas)
    # 举例用实际出现的标签，避免 PDF 的 [p.X] 格式串到 EPUB/音频/图片上
    if metas:
        _tags = [_cite_tag(metas[i]) for i in packed_idx if i < len(metas)]
        _uniq = list(dict.fromkeys(_tags))[:2]
        tag_example = " or ".join("[%s]" % t for t in _uniq) if _uniq else "[p.112]"
    else:
        tag_example = "[p.112]"
    out = _generate(LLM_MODEL,
                    PROMPT.format(context=context, question=question, tag_example=tag_example),
                    options={"temperature": TEMPERATURE, "num_predict": NUM_PREDICT})
```

> **注意**：`BRIEF_PROMPT` 和智能体的 `system_prompt`（第 466 行 `"- Always cite sources: text/figure as [p.X], audio as [audio mm:ss]."`）里也写死了 `[p.X]`，**同样会影响 EPUB**。如果要一并修，把那行改成：
> ```python
> "- Always cite sources by copying the bracket tag shown above each block exactly.\n"
> ```

---

## 修复二 · 升配闸门（省约 72% 无效升配）

### 问题

当前触发条件是「首答拒答就升配」：

```python
if dynamic and docs and is_abstain(answer):
```

它**无法区分「检索到了但预算不够」和「库里根本没有」**。实测 1873 次升配中 **1349 次（72%）花在不可答题上，升配后 97.8% 仍拒答**——纯浪费，且把拒答题的 token 中位数从 318 推到 742。

### 改动 1／3：加配置项（第 46–55 行的参数区）

```python
# 升配闸门：首答拒答后，若最优检索块的距离大于此值，判定为"库里真没有"，不再升配。
# None = 关闭闸门（旧行为，全部升配）。数值需先用 calib_gate.py 标定。
ESCALATE_SIM_GATE = None
```

### 改动 2／3：`ask()` 取出检索距离（第 683 行附近）

**原文：**

```python
def ask(col, question, verbose=True, dynamic=DYNAMIC_BUDGET):
    qv = embed([question])[0]
    res = col.query(query_embeddings=[qv], n_results=TOP_K)
    docs, metas = res["documents"][0], res["metadatas"][0]
```

**改成：**

```python
def ask(col, question, verbose=True, dynamic=DYNAMIC_BUDGET):
    qv = embed([question])[0]
    res = col.query(query_embeddings=[qv], n_results=TOP_K)
    docs, metas = res["documents"][0], res["metadatas"][0]
    _d = (res.get("distances") or [[]])[0]
    best_dist = min(_d) if _d else None      # 最优检索块的距离，越小越相关
```

### 改动 3／3：升配处加闸门（紧接着几行）

**原文：**

```python
    escalated = False
    # 动态升配：仅当"检索有命中却拒答"时，升到 1800 重答一次
    if dynamic and docs and is_abstain(answer):
```

**改成：**

```python
    escalated = False
    # 闸门：检索都不沾边就别升配了——实测 72% 的升配花在库外问题上且全部白跑
    _gate_ok = (ESCALATE_SIM_GATE is None or best_dist is None
                or best_dist <= ESCALATE_SIM_GATE)
    # 动态升配：仅当"检索有命中却拒答"且通过闸门时，升到 1800 重答一次
    if dynamic and docs and is_abstain(answer) and _gate_ok:
```

### 改动 4／4（可选但建议）：verbose 打印距离，便于标定

**原文：**

```python
        tag = "  (动态升配 %d)" % BUDGET_ESCALATED if escalated else ""
```

**改成：**

```python
        tag = "  (动态升配 %d)" % BUDGET_ESCALATED if escalated else ""
        if best_dist is not None:
            tag += "  dist=%.4f" % best_dist
```

---

## 标定阈值（必须先做，不能拍脑袋）

`ESCALATE_SIM_GATE` 设多少，取决于 ChromaDB 用的距离度量（默认 L2）和 bge-m3 的向量分布。**不能猜，要测。**

`calib_gate.py` 只做检索、不调 LLM，很快：

```powershell
E:
cd E:\Ollama_test

# 用一本已建好库的书标定（先确认 .built_book 是这本）
& "C:\Users\Seifer\distill\Scripts\python.exe" calib_gate.py `
    --main E:\Ollama_test\data\main.py --workdir E:\Ollama_test\data `
    --eval E:\Ollama_test\eval\eval_by_book --only Microbiology
```

它会输出可答题 vs 不可答题的**检索距离分布**，并给出一个使两者分离度最大的建议阈值，例如：

```
可答题   最优距离 中位 0.83 | P90 1.02
不可答题 最优距离 中位 1.24 | P10 1.05
建议 ESCALATE_SIM_GATE = 1.03
  → 预计拦掉 84% 的库外升配，误伤可答题 6%
```

**建议在 2–3 本不同学科的书上各标定一次**，取偏保守（偏大）的那个值——宁可少拦一些，不要误伤可答题。

---

## 验证顺序

```powershell
# 0. 存档基线（必做！改代码前）
Copy-Item eval_results eval_results_v1_baseline -Recurse -Force

# 1. 改完先单本验证（Microbiology，约 5 分钟）
& "C:\Users\Seifer\distill\Scripts\python.exe" run_eval_batch.py `
    --main E:\Ollama_test\data\main.py `
    --books E:\Ollama_test\books --eval E:\Ollama_test\eval\eval_by_book `
    --only Microbiology --out eval_results_v2

# 2. 再单验一本 EPUB（检查引用格式修好没）
& "C:\Users\Seifer\distill\Scripts\python.exe" run_eval_batch.py `
    --main E:\Ollama_test\data\main.py `
    --books E:\Ollama_test\books --eval E:\Ollama_test\eval\eval_by_book `
    --only pride-and-prejudice --out eval_results_v2

# 3. 都对再全量重跑（约 3 小时，改进后升配少了应该更快）
& "C:\Users\Seifer\distill\Scripts\python.exe" run_eval_batch.py `
    --main E:\Ollama_test\data\main.py --resume `
    --books E:\Ollama_test\books --eval E:\Ollama_test\eval\eval_by_book `
    --out eval_results_v2
```

**单本验证的预期**：

| 指标 | v1 基线 | v2 预期 |
|---|---|---|
| Microbiology 升配次数 | 35 | **明显下降**（不可答题不再升配） |
| Microbiology token 中位 | 408 | **下降** |
| 《傲慢与偏见》引用 `[p.N]` 越界 | 25 处 | **接近 0**，改为 `[ch:N]` |
| 可答命中率 | 89.7% / 75.9% | **不应下降**（若下降说明闸门卡太严） |

**判断标准**：可答命中率掉了 → 阈值调大；升配次数没降 → 阈值调小或没生效。
