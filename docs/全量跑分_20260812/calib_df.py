# -*- coding: utf-8 -*-
"""标定停用词阈值：词的**真实**命中块数（不受 limit 截断）。

为什么要重标：现行判据是 len(ids) >= int(N*ratio)，而取候选时带了 limit=POOL=40，
len(ids) 因此最大只有 40。只要库大于 200 块，int(N*0.2) > 40，条件恒假——
**过滤在几乎所有库上都是死代码**（已由 probe_kwfilter.py 证实）。

这里只测量、不改代码：用 include=[] 只取 ids，不设 limit，拿到真实 df，
再看"功能词碎片"与"真术语"能不能被一个比例分开。分不开就别改，
分得开就用数据定这个数——不拍脑袋。
"""
import os
import sys

sys.path.insert(0, r"E:\Ollama_test_beta\code")
import chromadb                                    # noqa: E402
import main as M                                   # noqa: E402
import webui                                       # noqa: E402

KB = r"E:\Ollama_test_beta\data\webui_knowledge_bases"
LIBS = [
    ("简明世界经济史(中)", "20260813_205105_d3ef435a"),
]

# 中文：真术语 vs 明显的功能词碎片（二元组切出来的）
CN_TERMS = ["五铢", "铢钱", "复式", "记账", "纸币", "汇票", "郁金", "金本", "殖民", "工业",
            "行了", "的发", "了多", "是谁", "什么", "以及", "这样", "因此", "可以", "的是"]


def df_of(col, term):
    """真实命中块数：只取 ids，不设 limit。"""
    try:
        got = col.get(where_document={"$contains": term}, include=[])
    except Exception as exc:
        return None, repr(exc)[:60]
    return len((got or {}).get("ids") or []), ""


for label, lid in LIBS:
    path = os.path.join(KB, lid, "vectordb")
    if not os.path.isdir(path):
        print("跳过（无此库）：%s" % label)
        continue
    col = chromadb.PersistentClient(path=path).get_or_create_collection(M.COLLECTION)
    n = col.count()
    print("\n===== %s：%d 块 =====" % (label, n))
    print("现行 df_max(0.2) = %d，而取候选 limit = %d → 判据恒假"
          % (webui._keyword_df_max(col), webui.HYBRID_KEYWORD_POOL))
    print("\n%-8s %8s %8s" % ("词", "真实df", "占比"))
    rows = []
    for t in CN_TERMS:
        d, err = df_of(col, t)
        if d is None:
            print("%-8s  取失败 %s" % (t, err)); continue
        rows.append((t, d, 100.0 * d / n))
        print("%-8s %8d %7.1f%%" % (t, d, 100.0 * d / n))
    real = [r for r in rows if r[0] in CN_TERMS[:10]]
    junk = [r for r in rows if r[0] in CN_TERMS[10:]]
    if real and junk:
        print("\n真术语 最大占比 %.1f%%（%s）" % (max(r[2] for r in real),
                                          max(real, key=lambda r: r[2])[0]))
        print("功能词 最小占比 %.1f%%（%s）" % (min(r[2] for r in junk),
                                          min(junk, key=lambda r: r[2])[0]))
        hi, lo = max(r[2] for r in real), min(r[2] for r in junk)
        print("→ %s" % ("可分：阈值取 %.0f%%–%.0f%% 之间都行" % (hi, lo) if lo > hi
                       else "**不可分**：两类占比区间重叠，单一比例阈值做不到"))
