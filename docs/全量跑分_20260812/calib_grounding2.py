# -*- coding: utf-8 -*-
"""重做接地率标定 —— 用**模型真实产出的改写句**，不用块内原句。

上一版取块里的整句当结论，接地率必然 1.0（逐字包含），中英都是 1.000，
测的是一件必然成立的事，毫无信息量。设计错了，重来。

这一版：拿真实答案的每个句子，对该题实际检索到的上下文算接地率，
中英同法，看两者的分布相对 _GROUNDED_MIN=0.3 各自站在哪。
中文样本来自 cn2_rows_default.jsonl 里答出来的题；英文来自 final_rows.jsonl 抽样。
"""
import io
import json
import os
import re
import sys

sys.path.insert(0, r"E:\Ollama_test_beta\code")
import main as M                                    # noqa: E402
import webui                                        # noqa: E402
from eval_compare import build_question_index, match_question_row  # noqa: E402

SP = os.path.dirname(os.path.abspath(__file__))
EVAL = r"E:\Ollama_test_beta\eval\eval_ALL.jsonl"
CITE = re.compile(r"\[[^\]]{1,80}\]")


def norm_name(n):
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", os.path.splitext(str(n or ""))[0]).lower()


reg = webui._read_registry()
byname = {}
for x in reg.get("libraries", []):
    for k in (x.get("name"), x.get("source")):
        if k:
            byname.setdefault(norm_name(k), x["id"])

eval_rows = [json.loads(line) for line in io.open(EVAL, encoding="utf-8") if line.strip()]
question_index = build_question_index(eval_rows)


def claims_of(answer):
    """按项目口径切句，并剥掉引用标签（标签不参与内容词重合）。"""
    text = CITE.sub(" ", answer or "")
    parts = M.split_sentences(text) if hasattr(M, "split_sentences") else re.split(r"[。！？.!?]\s*", text)
    return [p.strip() for p in parts if len(p.strip()) >= 8]


def measure(question, answer, lib_id):
    docs, metas, dists, targets = webui._retrieve_selected(question, [lib_id], False, None)
    ctx = "\n".join(docs)
    out = []
    for c in claims_of(answer):
        try:
            g = webui._grounding(c, ctx)
        except Exception:
            continue
        r = g[0] if isinstance(g, tuple) else g
        if isinstance(r, (int, float)):
            out.append(r)
    return out


def report(label, vals):
    if not vals:
        print("%-14s 无样本" % label); return
    vals.sort()
    lo = getattr(webui, "_GROUNDED_MIN", 0.3)
    under = sum(1 for v in vals if v < lo)
    q = lambda p: vals[min(len(vals) - 1, int(len(vals) * p))]
    print("%-14s n=%-4d 最小 %.3f  P10 %.3f  中位 %.3f  P90 %.3f   低于 %.2f 的 %d/%d = %.0f%%"
          % (label, len(vals), vals[0], q(.1), q(.5), q(.9), lo, under, len(vals),
             100.0 * under / len(vals)))


cn_vals = []
rows = [json.loads(l) for l in io.open(os.path.join(SP, "cn2_rows_default.jsonl"),
                                       encoding="utf-8") if l.strip()]
lid_cn = byname.get(norm_name("简明世界经济史"))
for r in rows:
    if r["outcome"] != "命中":
        continue
    cn_vals += measure(r["question"], r.get("answer") or "", lid_cn)

en_vals = []
erows = [json.loads(l) for l in io.open(os.path.join(SP, "reg_rows.jsonl"),
                                        encoding="utf-8") if l.strip()]
picked = [r for r in erows if r["outcome"] == "命中"][:20]
for r in picked:
    try:
        meta = match_question_row(r, question_index)
    except (KeyError, ValueError):
        continue
    lid = byname.get(norm_name(meta.get("book") or ""))
    if not lid:
        continue
    en_vals += measure(r["question"], r.get("answer") or "", lid)

print("_GROUNDED_MIN = %s（低于它判为无据、会被裁掉）\n" % getattr(webui, "_GROUNDED_MIN", "?"))
report("中文真实答案", cn_vals)
report("英文真实答案", en_vals)
