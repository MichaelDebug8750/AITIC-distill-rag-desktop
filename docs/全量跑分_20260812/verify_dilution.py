# -*- coding: utf-8 -*-
"""验证假设：证据下限看到的是被补充检索"稀释"过的距离，所以几乎拦不到东西。

结构上先推一遍（不用跑就能确定的部分）：
  _merge_retrieval 会把第一轮的 top-3 锁位保留，再并入第二轮结果。
  所以 min(合并后) ≤ min(第一轮) —— 合并只会让最优距离更小或持平。
而全做 Agent 之后，凡是模型答得出来的题都会跑第二轮；
编造恰恰全是"该拒答却答了"的题，也就是全都经过了合并。
=> 标定时我用 /api/retrieve 量的是单次检索距离，真实管线里闸门看到的比那更小。

这个脚本用真实数据量出两者的差，并算出：
  · 用合并后距离（现状）能拦住几条编造
  · 用第一轮距离（改法）能拦住几条编造
  · 两种做法各误杀多少正确答案
只用 /api/retrieve，不调模型。
"""
import io
import json
import os
import re
import statistics as st
import sys
import urllib.request

from eval_compare import build_question_index, match_question_row

SP = os.path.dirname(os.path.abspath(__file__))
EVAL = r"E:\Ollama_test_beta\eval\eval_ALL.jsonl"
B = "http://127.0.0.1:8011"
ROWS = os.path.join(SP, sys.argv[1] if len(sys.argv) > 1 else "final_rows.jsonl")
FLOOR = 0.99
TOP_K = 8
LOCKED = 3          # 与 _merge_retrieval 的锁位数一致


def norm(n):
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", os.path.splitext(str(n or ""))[0]).lower()


def retrieve(q, lib):
    body = {"question": q, "libraries": [lib], "top_k": TOP_K}
    rq = urllib.request.Request(B + "/api/retrieve", data=json.dumps(body).encode("utf-8"),
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(rq, timeout=300) as r:
            d = json.loads(r.read().decode("utf-8"))
    except Exception:
        return None
    return [(s.get("snippet"), s.get("distance")) for s in (d.get("sources") or [])]


def followup(q, rnd):
    """复刻 _followup_query 的行为：第 2/3 轮换个说法再检索。
       这里只需要"换过的查询"这一点，措辞不必逐字相同——
       要的是看合并会不会把最优距离拉低，而不是复现某一句。"""
    return "%s 定义 机制 条件 例子" % q if rnd == 2 else "%s 反例 限制 例外" % q


libs = json.loads(urllib.request.urlopen(B + "/api/libraries", timeout=180).read().decode("utf-8"))
libs = [x for x in (libs.get("libraries") or libs.get("items") or libs)
        if str(x.get("status") or "ready") == "ready"]
lid = {}
for x in libs:
    for k in (x.get("source"), x.get("name")):
        if k:
            lid.setdefault(norm(k), x.get("id"))

meta = build_question_index(
    [json.loads(line) for line in io.open(EVAL, encoding="utf-8") if line.strip()])

rows = [json.loads(l) for l in io.open(ROWS, encoding="utf-8") if l.strip()]
fab = [r for r in rows if r["outcome"] == "编造"]
# 代价侧必须量全：0/75 的 95%% 区间上界约 4%%，换算到 707 道可答题是 28 条，
# 这个不确定性太大，不能据此决策。测量只查检索不调模型，全量量得起。
good = [r for r in rows if r["outcome"] == "命中"]
print("样本：编造 %d 条，命中 %d 条\n" % (len(fab), len(good)))


def both_distances(r):
    m = match_question_row(r, meta)
    lib = lid.get(norm(m.get("book") or ""))
    if not lib:
        return None
    first = retrieve(r["question"], lib)
    if not first:
        return None
    d1 = [d for _s, d in first if isinstance(d, (int, float))]
    if not d1:
        return None
    round1_min = min(d1)
    # 模拟 _merge_retrieval：第一轮锁位 top-3，再并入第二轮
    second = retrieve(followup(r["question"], 2), lib) or []
    pool = first[:LOCKED] + [x for x in second if x[0] not in {s for s, _ in first[:LOCKED]}]
    d2 = [d for _s, d in pool[:TOP_K] if isinstance(d, (int, float))]
    merged_min = min(d2) if d2 else round1_min
    return round1_min, merged_min


def survey(rowset, label):
    r1, mg = [], []
    for r in rowset:
        got = both_distances(r)
        if not got:
            continue
        r1.append(got[0]); mg.append(got[1])
    if not r1:
        print(label, "无数据"); return None
    blocked_1 = sum(1 for d in r1 if d > FLOOR)
    blocked_m = sum(1 for d in mg if d > FLOOR)
    print("%s（n=%d）" % (label, len(r1)))
    print("   第一轮距离   中位 %.3f   >闸门 %d 条 = %.0f%%" % (st.median(r1), blocked_1, 100.0*blocked_1/len(r1)))
    print("   合并后距离   中位 %.3f   >闸门 %d 条 = %.0f%%" % (st.median(mg), blocked_m, 100.0*blocked_m/len(mg)))
    print("   合并把最优距离拉低了 %.3f（中位）" % (st.median(r1) - st.median(mg)))
    return blocked_1, blocked_m, len(r1)


f = survey(fab, "编造组（该拒答却答了）")
print()
g = survey(good, "命中组（正确作答，对照）")

if f and g:
    print()
    print("=== 换用第一轮距离的净效果 ===")
    print("   多拦住编造：%d 条" % (f[0] - f[1]))
    print("   多误杀正确：%d 条" % (g[0] - g[1]))
    print()
    if f[0] - f[1] <= 0:
        print("   结论：假设不成立，合并没有稀释掉拦截能力，不必改。")
    else:
        print("   结论：假设成立。但收益/代价必须和同条件空白对照比较，不能套固定噪声值。")
