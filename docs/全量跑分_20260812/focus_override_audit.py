# -*- coding: utf-8 -*-
"""全量离线审计：核心短语精确命中能否安全覆盖距离下限。"""
from __future__ import print_function

import argparse
import io
import json
import os
import re
import urllib.parse
import urllib.request
from collections import Counter

from eval_compare import normalize_book

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FLOOR = 0.99

_FOCUS_PATTERNS = [
    r"^(?:describe|define)\s+(.+?)$",
    r"^explain\s+the\s+term\s+(.+?)$",
    r"^give\s+the\s+definition\s+of\s+(.+?)$",
    r"^what\s+does\s+(?:the\s+term\s+)?(.+?)\s+(?:mean|refer\s+to)$",
    r"^what\s+is\s+meant\s+by\s+(.+?)$",
    r"^what\s+is\s+the\s+role\s+of\s+(.+?)$",
    r"^what\s+does\s+this\s+book\s+say\s+about\s+(.+?)$",
    r"^how\s+is\s+(.+?)\s+discussed\s+in\s+this\s+book$",
    r"^what\s+is\s+(?:an?\s+)?(.+?)$",
]


def words(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text or "").casefold()).strip()


def focus_phrase(question):
    cleaned = words(question)
    for pattern in _FOCUS_PATTERNS:
        match = re.match(pattern, cleaned, re.I)
        if not match:
            continue
        phrase = words(match.group(1))
        count = len(phrase.split())
        if 1 <= count <= 8 and len(phrase) >= 4:
            return phrase
    return ""


def fetch(base, path):
    with urllib.request.urlopen(base + path, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def load_jsonl(path):
    if not os.path.isfile(path):
        return []
    with io.open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("tag")
    args = parser.parse_args()
    base = "http://127.0.0.1:%s" % args.port

    eval_path = os.path.join(ROOT, "eval", "eval_ALL.jsonl")
    rows = load_jsonl(eval_path)
    payload = fetch(base, "/api/libraries")
    libraries = payload.get("libraries") or payload.get("items") or payload
    by_norm = {}
    for item in libraries:
        for value in (item.get("source"), item.get("name")):
            if value:
                by_norm.setdefault(normalize_book(value), item.get("id"))
    rows = [row for row in rows if normalize_book(row.get("book")) in by_norm]

    out_rows = os.path.join(HERE, "%s_rows.jsonl" % args.tag)
    prior = load_jsonl(out_rows)
    done = {(row["book_key"], row["question"]) for row in prior}
    for index, row in enumerate(rows, 1):
        key = normalize_book(row["book"])
        identity = (key, row["question"])
        if identity in done:
            continue
        query = urllib.parse.urlencode({"q": row["question"], "limit": 8})
        result = fetch(base, "/api/libraries/%s/chunks?%s" %
                       (urllib.parse.quote(str(by_norm[key]), safe=""), query))
        chunks = result.get("chunks") or []
        phrase = focus_phrase(row["question"])
        bodies = [words(item.get("text")) for item in chunks]
        phrase_hit = bool(phrase and any(phrase in body for body in bodies))
        best = min((item.get("distance") for item in chunks
                    if item.get("distance") is not None), default=None)
        record = {"book_key": key, "book": row["book"], "question": row["question"],
                  "type": row.get("type"), "expect": row.get("expect"),
                  "dataset_term": row.get("term"), "focus": phrase,
                  "focus_matches_term": bool(phrase and words(row.get("term")) == phrase),
                  "focus_hit": phrase_hit, "best_distance": best,
                  "floor_blocks": best is None or best >= FLOOR,
                  "chunks": chunks}
        with io.open(out_rows, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        prior.append(record); done.add(identity)
        if index % 50 == 0 or index == len(rows):
            print("%d/%d" % (index, len(rows)), flush=True)

    answerable = [row for row in prior if row.get("type") != "unanswerable"]
    probes = [row for row in prior if row.get("type") == "unanswerable"]
    blocked_answerable = [row for row in answerable if row["floor_blocks"]]
    blocked_probes = [row for row in probes if row["floor_blocks"]]
    summary = {
        "schema": 1, "tag": args.tag, "records": len(prior), "floor": FLOOR,
        "focus_extracted": sum(bool(row["focus"]) for row in prior),
        "focus_matches_dataset_term": sum(row["focus_matches_term"] for row in prior),
        "answerable": {"n": len(answerable),
                       "focus_hit": sum(row["focus_hit"] for row in answerable),
                       "blocked": len(blocked_answerable),
                       "blocked_focus_hit": sum(row["focus_hit"] for row in blocked_answerable)},
        "unanswerable": {"n": len(probes),
                         "focus_hit": sum(row["focus_hit"] for row in probes),
                         "blocked": len(blocked_probes),
                         "blocked_focus_hit": sum(row["focus_hit"] for row in blocked_probes)},
        "unanswerable_focus_collisions": [
            {key: row.get(key) for key in
             ("book", "question", "dataset_term", "focus", "best_distance")}
            for row in probes if row["focus_hit"]],
        "blocked_answerable_focus_hits": [
            {key: row.get(key) for key in
             ("book", "question", "dataset_term", "focus", "best_distance")}
            for row in blocked_answerable if row["focus_hit"]],
        "types": dict(Counter(row.get("type") for row in prior)),
    }
    out = os.path.join(HERE, "%s_analysis.json" % args.tag)
    temp = out + ".tmp"
    with io.open(temp, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    os.replace(temp, out)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
