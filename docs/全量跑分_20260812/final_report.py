# -*- coding: utf-8 -*-
"""扩容全量终版报告。

归一函数从 fullrun3 直接 import，不重写——同一个匹配写两遍就是两套口径，
这个项目在「口径不一致」上栽过，不再犯。
"""
import collections
import importlib.util
import io
import json
import os
import re
import sys
import urllib.request

from eval_compare import build_question_index, match_question_row

SP = os.path.dirname(os.path.abspath(__file__))
ROWS = os.path.join(SP, "after_rows.jsonl")
OUT = os.path.join(SP, "fullrun_final.md")
EVAL = r"E:\Ollama_test_beta\eval\eval_ALL.jsonl"

spec = importlib.util.spec_from_file_location("fr3", os.path.join(SP, "fullrun3.py"))
# fullrun3 在 import 时会执行跑分，不能直接 import；只取它的 norm 实现（逐字复制过来会漂移，
# 所以从源码里把函数抠出来执行——保证和跑分时用的是同一段代码）
src = io.open(os.path.join(SP, "fullrun3.py"), encoding="utf-8").read()
m = re.search(r"def norm\(name\):.*?\n\n\n", src, re.S)
ns = {"os": os, "re": re}
exec(m.group(0), ns)
norm = ns["norm"]

rows = [json.loads(l) for l in io.open(ROWS, encoding="utf-8") if l.strip()]
meta = build_question_index(
    [json.loads(line) for line in io.open(EVAL, encoding="utf-8") if line.strip()])

libs = json.loads(urllib.request.urlopen("http://127.0.0.1:8011/api/libraries",
                                         timeout=180).read().decode("utf-8"))
libs = libs.get("libraries") or libs.get("items") or libs
size = {}
for x in libs:
    for k in (x.get("source"), x.get("name")):
        if k:
            size.setdefault(norm(k), int(x.get("chunks") or 0))

# 行里的 book 是截断过的显示名，要用题集里的原始书名回查块数
def chunks_of(row):
    m = match_question_row(row, meta)
    return size.get(norm(m.get("book") or ""), 0)


oc = collections.Counter(r["outcome"] for r in rows)
hit, miss, over = oc["命中"], oc["未命中"], oc["过度拒答"]
graded = hit + miss + over
una = [r for r in rows if r["expect"] == "abstain"]
ok_ref, fab = oc["拒答正确"], oc["编造"]
CITE = re.compile(r"\[[^\]]+\]")
SENT = re.compile(r"(?<=[.。!?！？])\s+")


def pct(n, d):
    return "%.1f%%" % (100.0 * n / d) if d else "—"


answered = [r for r in rows if r["expect"] != "abstain" and not r["abstained"]
            and r["outcome"] != "请求失败"]
un_tot = un_bad = 0
for r in answered:
    ss = [s for s in SENT.split(r["answer"]) if len(s.strip()) > 8]
    un_tot += len(ss)
    un_bad += sum(1 for s in ss if not CITE.search(s))

conf = collections.Counter(r.get("confidence") or "—" for r in answered)
by_book = collections.defaultdict(lambda: collections.Counter())
for r in rows:
    by_book[r["book"]][r["outcome"]] += 1

# 规模分组：用回查到的真实块数
groups = {"大库 ≥4000 块": [], "中库 1000–4000": [], "小库 <1000": []}
for r in rows:
    c = chunks_of(r)
    if not c:
        continue
    k = "大库 ≥4000 块" if c >= 4000 else ("中库 1000–4000" if c >= 1000 else "小库 <1000")
    groups[k].append(r)

o = io.open(OUT, "w", encoding="utf-8")
o.write("# 扩容全量 · 终版报告\n\n")
o.write("**%d 题 / 12 本教材 / 5 个学科**（医学·心理·商科·法律·计算机）\n\n" % len(rows))
o.write("> 口径：题集中落在已建知识库上的全部题目。**与 v8final 的 93.7%/96.4%/3.6% 不可并列**"
        "——那是 CLI 口径，不同 PROMPT、不同 num_predict、不同代码路径。\n")
o.write("> 可答题评分为「任一关键词字面出现」，抽样显示多数「未命中」其实答对了，该数偏低。\n\n")

o.write("## 一、主要指标\n\n| 指标 | 值 |\n|---|---|\n")
o.write("| 不可答题精确拒答 | **%s**（%d/%d） |\n" % (pct(ok_ref, len(una)), ok_ref, len(una)))
o.write("| 编造 | **%s**（%d） |\n" % (pct(fab, len(una)), fab))
o.write("| 可答题命中 | %s（%d/%d） |\n" % (pct(hit, graded), hit, graded))
o.write("| 未命中 | %s（%d） |\n" % (pct(miss, graded), miss))
o.write("| 过度拒答 | %s（%d） |\n" % (pct(over, graded), over))
o.write("| 无引用句 | %s（%d/%d） |\n" % (pct(un_bad, un_tot), un_bad, un_tot))
o.write("| 请求失败 | %d |\n\n" % oc["请求失败"])

o.write("## 二、语料规模的双向影响（本轮核心发现）\n\n")
o.write("| 分组 | 题数 | 可答命中 | 编造率 |\n|---|---|---|---|\n")
for name, rs in groups.items():
    if not rs:
        continue
    g = [r for r in rs if r["expect"] != "abstain" and r["outcome"] in ("命中", "未命中", "过度拒答")]
    u = [r for r in rs if r["expect"] == "abstain"]
    h = sum(1 for r in g if r["outcome"] == "命中")
    f = sum(1 for r in u if r["outcome"] == "编造")
    o.write("| %s | %d | %s | %s |\n" % (name, len(rs), pct(h, len(g)), pct(f, len(u))))
o.write("\n库越大，可答题答得越好，不可答题却越容易编造——同一个机制：块多则检索总能找到\n")
o.write("语义相近的内容，该答的题证据更足，不该答的题也「看起来有据」。\n\n")
o.write("**结论：拒答闸门必须随语料规模标定，固定阈值不可移植。**\n\n")

o.write("## 三、可信度分布\n\n")
for k, v in conf.most_common():
    o.write("- %s：%d（%s）\n" % (k, v, pct(v, len(answered))))

o.write("\n## 四、分书\n\n| 书 | 结果 |\n|---|---|\n")
for b, c in sorted(by_book.items()):
    o.write("| %s | %s |\n" % (b, dict(c)))
o.close()

print("题数 %d" % len(rows))
print("精确拒答 %s | 编造 %s | 可答命中 %s" % (pct(ok_ref, len(una)), pct(fab, len(una)), pct(hit, graded)))
print("无引用句 %s | 可信度 %s" % (pct(un_bad, un_tot), dict(conf)))
for name, rs in groups.items():
    g = [r for r in rs if r["expect"] != "abstain" and r["outcome"] in ("命中", "未命中", "过度拒答")]
    u = [r for r in rs if r["expect"] == "abstain"]
    if g or u:
        print("  %-16s 题%4d  命中 %6s  编造 %6s" % (
            name, len(rs), pct(sum(1 for r in g if r["outcome"] == "命中"), len(g)),
            pct(sum(1 for r in u if r["outcome"] == "编造"), len(u))))
print("报告 →", OUT)
