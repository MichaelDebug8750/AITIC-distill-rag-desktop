# -*- coding: utf-8 -*-
"""真机冒烟：多库混合检索和前端同款 POST SSE 流式完成态。"""
from __future__ import print_function

import io
import json
import os
import sys
import urllib.request

from eval_compare import normalize_book

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "http://127.0.0.1:" + (sys.argv[1] if len(sys.argv) > 1 else "8021")
OUTPUT = os.path.join(HERE, "live_path_smoke_20260815.json")


def fetch_json(path):
    with urllib.request.urlopen(BASE + path, timeout=180) as response:
        return json.loads(response.read().decode("utf-8"))


def post(path, body, timeout=900):
    request = urllib.request.Request(BASE + path,
                                     data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.status, response.headers.get("Content-Type", ""), response.read()


def parse_sse(raw):
    events = []
    for block in raw.decode("utf-8").replace("\r\n", "\n").split("\n\n"):
        name = "message"
        data = []
        for line in block.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                data.append(line[5:].lstrip())
        if data:
            text = "\n".join(data)
            try:
                payload = json.loads(text)
            except ValueError:
                payload = {"raw": text}
            events.append({"event": name, "data": payload})
    return events


def main():
    payload = fetch_json("/api/libraries")
    libraries = payload.get("libraries") or payload.get("items") or payload
    by_norm = {}
    for item in libraries:
        if str(item.get("status") or "ready") != "ready":
            continue
        for value in (item.get("source"), item.get("name")):
            if value:
                by_norm.setdefault(normalize_book(value), item.get("id"))

    def find_library(name):
        key = normalize_book(name)
        exact = by_norm.get(key)
        if exact:
            return exact
        matches = {value for candidate, value in by_norm.items()
                   if key in candidate or candidate in key}
        return next(iter(matches)) if len(matches) == 1 else None

    think = find_library("Think Python")
    psych = find_library("Psychology2e_WEB")
    if not think or not psych:
        raise SystemExit("多库冒烟所需知识库不存在")

    multi_body = {
        "question": ("Compare what a stack diagram records with what reinforcement means "
                     "in operant conditioning."),
        "libraries": [think, psych], "mode": "auto", "style": "concise",
        "extend": False, "hybrid": True, "history": [],
    }
    multi_status, _multi_type, multi_raw = post("/api/ask", multi_body)
    multi = json.loads(multi_raw.decode("utf-8"))
    multi_ok = (multi_status == 200 and multi.get("retrieval") == "hybrid" and
                isinstance(multi.get("sources"), list))

    with io.open(os.path.join(HERE, "live_style_matrix_20260815.json"),
                 encoding="utf-8") as handle:
        matrix = json.load(handle)
    case = next(item for item in matrix["selected_cases"]
                if item["language"] == "English" and item["expected"] == "answer")
    library_id = by_norm.get(case["book_key"])
    if not library_id:
        raise SystemExit("流式冒烟稳定样本知识库不存在")
    stream_body = {"question": case["question"], "libraries": [library_id],
                   "mode": "auto", "style": "standard", "extend": False,
                   "hybrid": False, "history": []}
    stream_status, stream_type, stream_raw = post("/api/ask/stream", stream_body)
    events = parse_sse(stream_raw)
    event_names = [item["event"] for item in events]
    done = next((item["data"] for item in reversed(events) if item["event"] == "done"), None)
    stream_ok = (stream_status == 200 and "text/event-stream" in stream_type and
                 done is not None and "error" not in event_names)

    artifact = {
        "schema": 1, "base_url": BASE,
        "multi_library_hybrid": {"passed": multi_ok, "status": multi_status,
                                  "request": multi_body, "response": multi},
        "post_stream": {"passed": stream_ok, "status": stream_status,
                        "content_type": stream_type, "request": stream_body,
                        "event_names": event_names, "done": done},
        "passed": multi_ok and stream_ok,
    }
    temp = OUTPUT + ".tmp"
    with io.open(temp, "w", encoding="utf-8") as handle:
        json.dump(artifact, handle, ensure_ascii=False, indent=2)
    os.replace(temp, OUTPUT)
    print("multi-library hybrid pass=%s status=%s sources=%s" %
          (multi_ok, multi_status, len(multi.get("sources") or [])))
    print("POST stream pass=%s status=%s events=%s" %
          (stream_ok, stream_status, event_names))
    print("结果已写入 %s" % OUTPUT)
    return 0 if artifact["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
