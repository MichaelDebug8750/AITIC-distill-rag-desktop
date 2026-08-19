# -*- coding: utf-8 -*-
"""把 after_rows.jsonl 汇成一份可直接看的报告。

口径写死在这里，避免每次口头解释：
  · answerable / fuzzy_*  → 应作答；关键词命中即 hit，拒答记 over_refused
  · unanswerable          → 应输出精确 token [NO REFERENCE FOUND]；给实质答案记 fabricated
  · 与 v8final 的 CLI 数字不可并列（不同 PROMPT、不同 num_predict、不同路径）
"""
import collections
import io
import json
import os
import re
import sys

SP = os.path.dirname(os.path.abspath(__file__))
ROWS = os.path.join(SP, "after_rows.jsonl")
OUT = os.path.join(SP, "fullrun_report.md")
CITE = re.compile(r"\[[^\]]+\]")
SENT = re.compile(r"(?<=[.。!?！？])\s+")


def sentences(text):
    return [s for s in SENT.split(str(text or "")) if len(s.strip()) > 8]


rows = [json.loads(l) for l in io.open(ROWS, encoding="utf-8") if l.strip()]
if not rows:
    print("还没有数据"); sys.exit(0)

ans = [r for r in rows if r["expect"] != "abstain"]
una = [r for r in rows if r["expect"] == "abstain"]
oc = collections.Counter(r["outcome"] for r in rows)
hit = oc["命中"]; miss = oc["未命中"]; over = oc["过度拒答"]
graded = hit + miss + over
ok_ref = oc["拒答正确"]; fab = oc["编造"]; failed = oc["请求失败"]


def pct(n, d):
    return "%.1f%%" % (100.0 * n / d) if d else "—"


# 无引用句：本轮的核心问题
answered = [r for r in ans if not r["abstained"] and r["outcome"] != "请求失败"]
un_tot = un_bad = 0
per_q = []
for r in answered:
    ss = sentences(r["answer"])
    bad = sum(1 for s in ss if not CITE.search(s))
    un_tot += len(ss); un_bad += bad
    if ss:
        per_q.append((bad / len(ss), r))

conf = collections.Counter(r.get("confidence") or "—" for r in answered)
rounds = collections.Counter(r.get("rounds") or 0 for r in rows)
pruned = sum(int(r.get("pruned") or 0) for r in rows)
orphan = sum(int(r.get("orphaned") or 0) for r in rows)
unknown = sum(int(r.get("unknown") or 0) for r in rows)
by_book = collections.defaultdict(collections.Counter)
for r in rows:
    by_book[r["book"]][r["outcome"]] += 1

o = io.open(OUT, "w", encoding="utf-8")
o.write("# webui 路径全量 · 结果报告\n\n")
o.write("> 口径：题集中落在已建知识库上的全部题目（Think Python / Dreams / Criminal Law）。\n")
o.write("> **与 v8final 的 93.7%/96.4%/3.6% 不可并列**——那是 CLI 口径，不同 PROMPT、"
        "不同 num_predict、不同代码路径。\n")
o.write("> **耗时数据作废**：本次机器 GPU 锁频 225MHz/7W（约正常 1/30）。\n\n")
o.write("已完成 **%d / 196** 题。\n\n" % len(rows))

o.write("## 一、主要指标\n\n| 指标 | 值 |\n|---|---|\n")
o.write("| 可答题命中率 | %s（%d/%d） |\n" % (pct(hit, graded), hit, graded))
o.write("| 可答题未命中 | %s（%d） |\n" % (pct(miss, graded), miss))
o.write("| 过度拒答 | %s（%d） |\n" % (pct(over, graded), over))
o.write("| 不可答题精确拒答 | %s（%d/%d） |\n" % (pct(ok_ref, len(una)), ok_ref, len(una)))
o.write("| 编造（不可答题给了实质答案） | %s（%d） |\n" % (pct(fab, len(una)), fab))
if failed:
    o.write("| 请求失败 | %d |\n" % failed)
o.write("\n")

o.write("## 二、无引用句（本轮核心问题）\n\n")
if un_tot:
    o.write("已作答的 %d 题共 %d 句，其中 **%d 句没有引用（%s）**。\n\n"
            % (len(answered), un_tot, un_bad, pct(un_bad, un_tot)))
    o.write("界面副标题承诺「每句话可溯源到原文页码」；无引用句会进逐句核验、"
            "多数判不出来，从而把可信度拖到「低」。\n\n")
    # 只按占比排；占比相同时元组会退到比较 dict → TypeError
    per_q.sort(key=lambda x: x[0], reverse=True)
    o.write("无引用占比最高的 8 题：\n\n| 占比 | 书 | 问题 |\n|---|---|---|\n")
    for ratio, r in per_q[:8]:
        o.write("| %.0f%% | %s | %s |\n" % (ratio * 100, r["book"], r["question"][:60]))
    o.write("\n")
else:
    o.write("尚无已作答题目。\n\n")

o.write("## 三、可信度分布\n\n")
for k, v in conf.most_common():
    o.write("- %s：%d（%s）\n" % (k, v, pct(v, len(answered))))
o.write("\n## 四、Agent 与逐句核验\n\n")
o.write("- 轮次分布：%s\n" % dict(rounds))
o.write("- 逐句核验累计：裁剪 %d ｜ 悬空剔除 %d ｜ 未判定 %d\n\n" % (pruned, orphan, unknown))

o.write("## 五、分书\n\n| 书 | 结果 |\n|---|---|\n")
for book, cnt in by_book.items():
    o.write("| %s | %s |\n" % (book, dict(cnt)))

bad_rows = [r for r in rows if r["outcome"] in ("编造", "过度拒答", "未命中", "请求失败")]
o.write("\n## 六、需人工复核的题（%d 条）\n\n" % len(bad_rows))
for r in bad_rows[:30]:
    o.write("- **%s** ｜ %s ｜ %s\n  - 答案：%s\n"
            % (r["outcome"], r["book"], r["question"][:70],
               str(r["answer"])[:160].replace("\n", " ")))
o.close()

print("已完成 %d/196" % len(rows))
print("可答命中 %s | 精确拒答 %s | 编造 %s" % (pct(hit, graded), pct(ok_ref, len(una)), pct(fab, len(una))))
print("无引用句 %s（%d/%d）" % (pct(un_bad, un_tot), un_bad, un_tot))
print("可信度分布:", dict(conf))
print("报告 →", OUT)
