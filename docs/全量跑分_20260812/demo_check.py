# -*- coding: utf-8 -*-
"""演示三题在一个显式配置下的实际表现，并把原始结果落盘。

聚合指标再好，现场问的是具体这几道。用法：
  demo_check.py <标签> [port] --hybrid on|off [--output 路径]
"""
import argparse
import io
import json
import os
import re
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
parser = argparse.ArgumentParser()
parser.add_argument("tag")
parser.add_argument("port", nargs="?", default="8011")
parser.add_argument("--hybrid", choices=("on", "off"), required=True)
parser.add_argument("--output")
args = parser.parse_args()

B = "http://127.0.0.1:" + args.port
TAG = args.tag
HYBRID = args.hybrid == "on"
OUTPUT = os.path.abspath(args.output or os.path.join(HERE, "%s_demo.json" % TAG))
CITE = re.compile(r"\[[^\]]+\]")

CASES = [
    ("Think Python", "What is a stack diagram, and what information does each frame contain?"),
    ("The Interpretation of Dreams", "What is dream-content?"),
    ("Criminal Law", "How does the book distinguish actus reus from mens rea, "
                     "and why are both generally required?"),
]

libs = json.loads(urllib.request.urlopen(B + "/api/libraries", timeout=180).read().decode("utf-8"))
libs = [x for x in (libs.get("libraries") or libs.get("items") or libs)
        if x.get("status") == "ready"]


def find(k):
    for x in libs:
        if k.lower() in str(x.get("name") or "").lower():
            return x["id"]


status = json.loads(urllib.request.urlopen(B + "/api/status", timeout=180).read().decode("utf-8"))
results = []
print("=== 演示三题 · 配置 %s · hybrid=%s ===" % (TAG, HYBRID))
for book, q in CASES:
    lid = find(book)
    item = {"book": book, "question": q, "library_id": lid,
            "hybrid_requested": HYBRID}
    if not lid:
        item["error"] = "找不到库"
        results.append(item)
        print("  %-30s 找不到库" % book[:28]); continue
    body = {"question": q, "libraries": [lid], "mode": "auto",
            "style": "standard", "extend": True, "hybrid": HYBRID, "history": []}
    rq = urllib.request.Request(B + "/api/ask", data=json.dumps(body).encode("utf-8"),
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(rq, timeout=900) as r:
            d = json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        item["error"] = "%s: %s" % (type(exc).__name__, exc)
        results.append(item)
        print("  %-30s 请求失败 %r" % (book[:28], exc)); continue
    ag = d.get("agent") or {}
    au = ag.get("support_audit") or {}
    ans = d.get("answer") or ""
    sup = (d.get("supplement") or {}).get("text") or ""
    print("  %-28s %5.1fs 轮次=%s 拒答=%s 引用校验=%s 可信度=%s"
          % (book[:26], time.time() - t0, ag.get("rounds"), d.get("abstained"),
             (d.get("cite_check") or {}).get("ok"), (ag.get("confidence") or {}).get("level")))
    print("      教材依据 %d 字 / 引用 %d 处   完整解答 %d 字   裁剪=%s 未判定=%s"
          % (len(ans), len(CITE.findall(ans)), len(sup),
             au.get("pruned"), au.get("unknown")))
    print("      来源: %s" % [s.get("label") for s in (d.get("sources") or [])][:3])
    item.update({"status": 200, "elapsed": round(time.time() - t0, 3),
                 "abstained": bool(d.get("abstained")), "answer": ans,
                 "supplement": sup, "retrieval": d.get("retrieval"),
                 "cite_ok": (d.get("cite_check") or {}).get("ok"),
                 "confidence": (ag.get("confidence") or {}).get("level"),
                 "rounds": ag.get("rounds"), "sources": d.get("sources") or [],
                 "support_audit": au})
    results.append(item)

artifact = {"schema": 1, "tag": TAG, "base_url": B,
            "hybrid_requested": HYBRID, "service": status, "results": results}
temp = OUTPUT + ".tmp"
with io.open(temp, "w", encoding="utf-8") as handle:
    json.dump(artifact, handle, ensure_ascii=False, indent=2)
os.replace(temp, OUTPUT)
print("结果已写入 %s" % OUTPUT)
if any(item.get("error") for item in results) or len(results) != len(CASES):
    raise SystemExit(1)
