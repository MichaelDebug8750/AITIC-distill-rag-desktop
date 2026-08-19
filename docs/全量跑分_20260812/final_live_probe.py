# -*- coding: utf-8 -*-
"""最终代码的小规模真机行为抽查；不冒充全量统计。"""
import io
import json
import os
import time
import urllib.error
import urllib.request

BASE = "http://127.0.0.1:8011"
HERE = os.path.dirname(os.path.abspath(__file__))
OUTPUT = os.path.join(HERE, "final_live_probe_20260814.jsonl")
NO_REF = "[NO REFERENCE FOUND]"

CASES = [
    ("Principles of Microeconomics", "20260812_142932_8eb4a565",
     "What does federal system mean?", "abstain", []),
    ("Operating Systems - Three Easy Pieces", "20260812_143248_af2845ed",
     "What is meant by favorability poll?", "abstain", []),
    ("Psychology2e_WEB", "20260812_142732_44c49025",
     "Describe lifetime ban.", "abstain", []),
    ("The Interpretation of Dreams", "legacy",
     "Define majority party.", "abstain", []),
    ("Psychology2e_WEB", "20260812_142732_44c49025",
     "optimize branch studies", "answer", ["human factors psychology"]),
    ("Concepts of Biology", "20260812_142632_8a406ffc",
     "masks versions describes", "answer", ["dominant"]),
]


def ask(question, library_id):
    body = {
        "question": question, "libraries": [library_id], "mode": "auto",
        "style": "standard", "extend": False, "history": [], "hybrid": True,
    }
    request = urllib.request.Request(
        BASE + "/api/ask", data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, {"error": exc.read().decode("utf-8", "replace")[:500]}
    except Exception as exc:
        return 0, {"error": "%s: %s" % (type(exc).__name__, str(exc)[:300])}


rows = []
for book, library_id, question, expect, keywords in CASES:
    started = time.time()
    status, result = ask(question, library_id)
    answer = str(result.get("answer") or "").strip()
    abstained = bool(result.get("abstained"))
    if expect == "abstain":
        passed = status == 200 and abstained and answer == NO_REF
    else:
        passed = (status == 200 and not abstained
                  and any(keyword.lower() in answer.lower() for keyword in keywords))
    row = {
        "book": book, "library_id": library_id, "question": question,
        "expect": expect, "keywords": keywords, "status": status, "passed": passed,
        "abstained": abstained, "retrieval": result.get("retrieval"),
        "rounds": (result.get("agent") or {}).get("rounds"),
        "stop_reason": (result.get("agent") or {}).get("stop_reason"),
        "elapsed": round(time.time() - started, 1), "answer": answer,
        "error": result.get("error"),
    }
    rows.append(row)
    print("%s | %s | %s | %.1fs" %
          ("PASS" if passed else "FAIL", expect, question, row["elapsed"]), flush=True)

with io.open(OUTPUT, "w", encoding="utf-8") as handle:
    for row in rows:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
print("SUMMARY %d/%d passed; wrote %s" %
      (sum(row["passed"] for row in rows), len(rows), OUTPUT))
