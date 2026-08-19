# -*- coding: utf-8 -*-
"""真机验证：当前构建 + 混合检索，中文库外题还会不会被打穿。

cn2h 那一臂（昨晚 21:21）是 0/10 精确拒答，但它跑在 _usable_dists 修复之前——
那时含 None 的距离进 should_escalate 会抛 TypeError。静态探针显示当前构建
两条路都 拦=True，所以怀疑崩溃修复顺带修好了这个问题。

打 /api/ask 实测，不靠推断。用法：probe_cn_live.py [port]
"""
import io
import json
import os
import sys
import time
import urllib.request

SP = os.path.dirname(os.path.abspath(__file__))
B = "http://127.0.0.1:" + (sys.argv[1] if len(sys.argv) > 1 else "8011")

libs = json.loads(urllib.request.urlopen(B + "/api/libraries", timeout=180).read().decode("utf-8"))
libs = libs.get("libraries") or libs.get("items") or libs
lid = None
for x in libs:
    if "简明世界经济史" in str(x.get("name") or ""):
        lid = x.get("id")
if not lid:
    raise SystemExit("服务里没有这个库")

rows = [json.loads(l) for l in io.open(os.path.join(SP, "eval_cn2.jsonl"), encoding="utf-8")
        if l.strip()]
una = [r for r in rows if r["type"] == "unanswerable"]
print("库外题 %d 道，当前构建 + 混合检索\n" % len(una))

ok = fab = 0
for r in una:
    body = {"question": r["question"], "libraries": [lid], "mode": "auto",
            "style": "standard", "extend": False, "history": []}
    rq = urllib.request.Request(B + "/api/ask", data=json.dumps(body).encode("utf-8"),
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(rq, timeout=900) as resp:
            d = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print("  %-24s 请求失败 %r" % (r["question"][:22], exc)); continue
    ans = (d.get("answer") or "").strip()
    ab = bool(d.get("abstained"))
    exact = ans == "[NO REFERENCE FOUND]"
    if ab:
        ok += 1
    else:
        fab += 1
    print("  %-24s 拒答=%-5s 逐字契约=%-5s %.1fs  %s"
          % (r["question"][:22], ab, exact, time.time() - t0,
             "" if ab else ans[:44].replace("\n", " ")))

print("\n精确拒答 %d/%d，编造 %d" % (ok, len(una), fab))
print("对照：cn2h（修复前，混合）0/10 ；cn2（纯向量）10/10")
