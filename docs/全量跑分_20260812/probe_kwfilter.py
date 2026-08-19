# -*- coding: utf-8 -*-
"""查证：停用词过滤在中文库上到底触没触发。

疑点：col.get(..., limit=POOL) 会把 len(ids) 截断在 POOL=40，
而 218 块库的 df_max = int(218*0.2) = 43。若判据是 len(ids) >= df_max，
则 40 >= 43 永远为假 —— 过滤根本不会触发，cn2kw 应与 cn2h 完全相同。
可实测两臂差了 0/10 vs 10/10 的拒答。必须查清楚是哪一头错了。
"""
import os
import sys

sys.path.insert(0, r"E:\Ollama_test_beta\code")
import chromadb                                    # noqa: E402
import main as M                                   # noqa: E402
import webui                                       # noqa: E402

P = r"E:\Ollama_test_beta\data\webui_knowledge_bases\20260813_205105_d3ef435a\vectordb"
col = chromadb.PersistentClient(path=P).get_or_create_collection(M.COLLECTION)
dfm = webui._keyword_df_max(col)
print("库块数 = %d   POOL = %d   df_max = %s" % (col.count(), webui.HYBRID_KEYWORD_POOL, dfm))

QS = [
    "什么是心肌梗死的典型心电图表现？",
    "简述孟德尔遗传定律的核心内容。",
    "五铢钱是谁规定的？发行了多少？",
]
for q in QS:
    terms = webui._query_terms(q)
    print("\nQ: %s" % q)
    print("  切词: %s" % terms[:8])
    for t in terms[:8]:
        try:
            got = col.get(where_document={"$contains": t},
                          limit=webui.HYBRID_KEYWORD_POOL, include=["documents"])
        except Exception as exc:
            print("   %-6s 取候选失败 %r" % (t, exc)); continue
        n = len((got or {}).get("ids") or [])
        print("   %-6s 命中 %2d 块 %s" % (t, n, "<- 达到阈值，被过滤" if (dfm and n >= dfm) else ""))
    on = webui._keyword_rank(col, q)
    saved = os.environ.get("AITIC_KW_DF_RATIO")
    os.environ["AITIC_KW_DF_RATIO"] = "0"
    off = webui._keyword_rank(col, q)
    if saved is None:
        os.environ.pop("AITIC_KW_DF_RATIO", None)
    else:
        os.environ["AITIC_KW_DF_RATIO"] = saved
    same = [d for d, _ in on] == [d for d, _ in off]
    print("  过滤开 %d 块 / 过滤关 %d 块 / 完全相同 = %s" % (len(on), len(off), same))
