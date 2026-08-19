# -*- coding: utf-8 -*-
"""混合检索的收益与代价，逐条看原文再下结论。

净值 +36 是按"1 条编造 = 2 条正确答案"算的，而这个兑换率是我自己设的。
在把它交给用户拍板之前，先把两侧的**成色**看清楚：

  代价侧：拒答正确 → 编造，是真从材料外编的，还是"先声明没有再作答"那种
          （§三十 已证实那种形态确实是编造，但要逐条确认，不能套用）
  收益侧：过度拒答 → 命中，是真答对了，还是关键词凑巧命中了评分词

判分器只看"任一关键词字面出现"，所以两侧都可能虚高/虚低，必须读原文。
"""
import io
import json
import os
import re

from eval_compare import build_question_index, load_rows, match_question_row

SP = os.path.dirname(os.path.abspath(__file__))


base = load_rows(os.path.join(SP, "reg_rows.jsonl"))
hyb = load_rows(os.path.join(SP, "hyb_rows.jsonl"))
common = sorted(set(base) & set(hyb))

# 跑分结果里没有 keywords 字段（它在题集里），第一版脚本直接 r.get("keywords")
# 拿到的永远是 None，于是"命中词"全空、统计毫无意义。从题集补进来。
EVAL = r"E:\Ollama_test_beta\eval\eval_ALL.jsonl"
eval_index = build_question_index(
    [json.loads(line) for line in io.open(EVAL, encoding="utf-8") if line.strip()])


def keywords_for(key):
    return match_question_row(base[key], eval_index).get("keywords") or []

cost = [q for q in common if base[q]["outcome"] == "拒答正确" and hyb[q]["outcome"] == "编造"]
gain = [q for q in common if base[q]["outcome"] == "过度拒答" and hyb[q]["outcome"] == "命中"]

# "先声明材料里没有、再接着答"的形态（§三十 已判定这仍属编造，此处只做分类统计）
HEDGE = re.compile(
    r"(not\s+(directly\s+)?(mentioned|addressed|defined|discussed|provided|covered|found)"
    r"|does\s+not\s+(mention|address|define|discuss|provide|contain)"
    r"|no\s+(mention|reference|information)\s+of)", re.I)

print("=== 代价侧：拒答正确 → 编造，共 %d 条（全列）===\n" % len(cost))
hedged = 0
for q in cost:
    a = (hyb[q].get("answer") or "").strip()
    h = bool(HEDGE.search(a[:260]))
    hedged += 1 if h else 0
    print("  Q: %s" % q[1][:66])
    print("     %s%s" % ("[先声明后作答] " if h else "[直接给内容] ", a[:150].replace("\n", " ")))
    print()
print("  形态分布：先声明后作答 %d / 直接给内容 %d" % (hedged, len(cost) - hedged))

print("\n=== 收益侧：过度拒答 → 命中，共 %d 条（抽 8 条看成色）===\n" % len(gain))
for q in gain[:8]:
    r = hyb[q]
    a = (r.get("answer") or "").strip()
    kws = keywords_for(q)
    hit = [k for k in kws if k.lower() in a.lower()]
    print("  Q: %s" % q[1][:66])
    print("     命中词 %s" % hit)
    print("     %s" % a[:150].replace("\n", " "))
    print()

print("=== 收益侧整体：命中词是不是只擦到一个边 ===")
one_kw = 0
for q in gain:
    a = (hyb[q].get("answer") or "").lower()
    hit = [k for k in keywords_for(q) if k.lower() in a]
    if len(hit) <= 1:
        one_kw += 1
print("  %d / %d 条只命中 1 个关键词（越多说明判分越勉强）" % (one_kw, len(gain)))

# 反向核对代价侧：这些"编造"的答案里，有没有把问句里的词答成了另一个概念
print("\n=== 代价侧的形态：是不是词形碰撞 ===")
print("  （问句术语与答案主题明显不是一回事，即 BM25 按字面拉回了无关块）")
COLLISION = [
    ("circuit courts", "circuitous"), ("data structure", "nucleotide"),
    ("divergent thinking", "divergent evolution"), ("anti-dumping", "ocean dumping"),
    ("majority rule", "majority of species"), ("prior restraint", "restraint in harvesting"),
]
for term, wrong in COLLISION:
    for q in cost:
        if term.lower() in q[1].lower():
            a = (hyb[q].get("answer") or "").lower()
            print("  %-22s → 答案里出现 %-24s %s"
                  % (term, wrong, "是" if wrong.lower() in a else "否"))
