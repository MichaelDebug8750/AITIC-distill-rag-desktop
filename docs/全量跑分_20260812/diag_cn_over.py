# -*- coding: utf-8 -*-
"""诊断：简明世界经济史 20 道可答题里，11 道被过度拒答，死在哪一环。

英文侧的同类诊断（§二十九）结论是：107 条全部死在校验轮、80% 的检索距离
其实很好。中文这本是不是同一个病灶，还是另有原因（比如检索根本没召回）。

不跑模型，只读跑分结果 + 一次检索。
"""
import io
import json
import os
import re
import sys

sys.path.insert(0, r"E:\Ollama_test_beta\code")
import webui                                        # noqa: E402

SP = os.path.dirname(os.path.abspath(__file__))
rows = [json.loads(l) for l in io.open(os.path.join(SP, "cn2_rows_default.jsonl"),
                                       encoding="utf-8") if l.strip()]
ev = {r["question"]: r for r in
      (json.loads(l) for l in io.open(os.path.join(SP, "eval_cn2.jsonl"), encoding="utf-8")
       if l.strip())}

over = [r for r in rows if r["outcome"] == "过度拒答"]
hit = [r for r in rows if r["outcome"] == "命中"]
print("过度拒答 %d 道 / 命中 %d 道\n" % (len(over), len(hit)))

print("=== 死在第几轮 / stop_reason ===")
for r in over:
    print("  轮次=%s  %-30s %s" % (r.get("rounds"), r["question"][:28],
                                  str(r.get("stop_reason"))[:38]))

reg = webui._read_registry()
lid = None
for x in reg.get("libraries", []):
    if x.get("name") == "简明世界经济史":
        lid = x["id"]
if not lid:
    raise SystemExit("找不到库")

print("\n=== 检索侧：证据到底在不在（用题集里人工核过的 evidence 比对）===")
print("%-28s %-8s %-8s %s" % ("问题", "最优距离", "闸门拦?", "证据块是否被召回"))


def norm(s):
    return re.sub(r"\s+", "", str(s or ""))


for r in over:
    q = r["question"]
    e = norm((ev.get(q) or {}).get("evidence"))
    docs, metas, dists, targets = webui._retrieve_selected(q, [lid], False, None)
    usable = webui._usable_dists(dists)
    blocked = webui._evidence_floor_blocks(dists)
    found = any(e and e[:24] in norm(d) for d in docs)
    print("%-28s %-8s %-8s %s"
          % (q[:26], ("%.3f" % min(usable)) if usable else "无", blocked,
             "在 top-8 里" if found else "**没召回**"))
