# -*- coding: utf-8 -*-
"""改探真实路径：_retrieve_selected（中文库不是 M.DB_PATH，走多库分支）。

上一版探针直接调 _retrieve_hybrid，结论是"两条路判决相同、闸门都拦住"，
与实测 0/10 矛盾——因为真实请求根本不走那个函数。
这一版按 api_ask 的调用方式来。
"""
import io
import json
import os
import sys

sys.path.insert(0, r"E:\Ollama_test_beta\code")
import webui                                       # noqa: E402

SP = os.path.dirname(os.path.abspath(__file__))
LIB_NAME = "简明世界经济史"

reg = webui._read_registry()
lid = None
for x in reg.get("libraries", []):
    if x.get("name") == LIB_NAME:
        lid = x["id"]
        break
if not lid:
    raise SystemExit("找不到库：%s" % LIB_NAME)
print("库 id = %s" % lid)
print("活动库 M.DB_PATH = %s" % os.path.basename(os.path.dirname(webui.M.DB_PATH)))
print("→ 中文库%s活动库，所以%s单库快路\n"
      % ("就是" if lid in webui.M.DB_PATH else "不是",
         "走" if lid in webui.M.DB_PATH else "不走"))

rows = [json.loads(l) for l in io.open(os.path.join(SP, "eval_cn2.jsonl"), encoding="utf-8")
        if l.strip()]
una = [r for r in rows if r["type"] == "unanswerable"]

for label, hybrid in (("纯向量", False), ("混合", True)):
    print("===== %s =====" % label)
    blocked = 0
    for r in una:
        docs, metas, dists, targets = webui._retrieve_selected(r["question"], [lid], hybrid, None)
        usable = webui._usable_dists(dists)
        n_none = sum(1 for d in (dists or []) if d is None)
        blk = webui._evidence_floor_blocks(dists)
        rich = webui._evidence_looks_present(dists)
        blocked += 1 if blk else 0
        print("  %-24s min=%-7s None=%d/%-2d 拦=%-5s 讲解档=%s"
              % (r["question"][:22],
                 ("%.3f" % min(usable)) if usable else "无",
                 n_none, len(dists or []), blk, rich))
    print("  → 10 道里闸门拦住 %d 道\n" % blocked)
