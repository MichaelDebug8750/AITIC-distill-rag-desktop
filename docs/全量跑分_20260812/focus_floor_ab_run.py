# -*- coding: utf-8 -*-
"""焦点短语证据下限的同服务逐题配对 A/B；支持断点恢复和构建指纹。

A 始终显式关闭 ``_focus_floor``，B 始终显式开启；两臂都显式关闭 hybrid。
每题相邻运行，第二遍反转题序和臂序，避免把跨时段漂移误算成改动收益。
"""
from __future__ import print_function

import argparse
import hashlib
import io
import json
import os
import platform
import random
import sys
import time

from paired_ab_run import (fetch_json, library_snapshot, load_suite, norm, score,
                           sha256, stable_status)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CODE = os.path.join(ROOT, "code")


def call(base, row, library_id, focus_floor, attempts=3):
    """调用问答端点；实验开关和混合检索均显式传入，禁止依赖默认值。"""
    import urllib.error
    import urllib.request

    body = {"question": row["question"], "libraries": [library_id], "mode": "auto",
            "style": "standard", "extend": False, "hybrid": False,
            "_focus_floor": bool(focus_floor), "history": []}
    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
    last = (0, {"error": "request not attempted"})
    for attempt in range(attempts):
        request = urllib.request.Request(base + "/api/ask", data=raw,
                                         headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", "replace")
            try:
                payload = json.loads(text or "{}")
            except ValueError:
                payload = {"error": text[:500] or "HTTP %d" % exc.code}
            last = (exc.code, payload)
            if exc.code < 500 or attempt + 1 >= attempts:
                return last
        except Exception as exc:
            last = (0, {"error": "%s: %s" % (type(exc).__name__, str(exc)[:300])})
            if attempt + 1 >= attempts:
                return last
        time.sleep(min(2 ** attempt, 4))
    return last


def fingerprints(paths):
    targets = list(paths) + [
        os.path.join(CODE, "webui.py"), os.path.join(CODE, "webui_index.html"),
        os.path.join(CODE, "test_pipeline.py"), os.path.join(CODE, "main.py"),
        os.path.join(HERE, "paired_ab_run.py"), __file__,
    ]
    return {os.path.relpath(path, ROOT): sha256(path) for path in targets}


def service_snapshot(status):
    result = stable_status(status)
    result["focus_floor_default"] = status.get("focus_floor_default")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("tag")
    parser.add_argument("--suite", choices=("english", "chinese"), required=True)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260816)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.passes <= 4:
        raise SystemExit("passes 必须为 1..4")

    base = "http://127.0.0.1:%s" % args.port
    paths, rows = load_suite(args.suite)
    status = fetch_json(base + "/api/status")
    if not status.get("ready"):
        raise SystemExit("服务未 ready：%r" % status)
    if status.get("focus_floor_default") is not False:
        raise SystemExit("服务的 focus_floor_default 必须为 false，实际=%r" %
                         status.get("focus_floor_default"))
    library_payload = fetch_json(base + "/api/libraries")
    libraries = library_payload.get("libraries") or library_payload.get("items") or library_payload
    libraries = [item for item in libraries if str(item.get("status") or "ready") == "ready"]
    by_norm = {}
    for item in libraries:
        for value in (item.get("source"), item.get("name")):
            if value:
                by_norm.setdefault(norm(value), item.get("id"))
    missing = sorted({row["book"] for row in rows if norm(row["book"]) not in by_norm})
    if missing:
        if args.suite == "chinese":
            raise SystemExit("中文题集缺知识库：%s" % missing)
        rows = [row for row in rows if norm(row["book"]) in by_norm]

    row_path = os.path.join(HERE, "%s_rows.jsonl" % args.tag)
    manifest_path = os.path.join(HERE, "%s_manifest.json" % args.tag)
    manifest = {
        "schema": 1, "experiment": "focus_floor", "tag": args.tag,
        "suite": args.suite, "port": args.port, "passes": args.passes,
        "seed": args.seed, "expected_cases": len(rows),
        "expected_records": len(rows) * args.passes * 2,
        "request_arms": {"A": False, "B": True}, "hybrid_requested": False,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "python": platform.python_version(), "service": service_snapshot(status),
        "libraries": library_snapshot(libraries), "fingerprints": fingerprints(paths),
    }
    identity_keys = ("schema", "experiment", "suite", "passes", "seed",
                     "expected_cases", "expected_records", "request_arms",
                     "hybrid_requested", "service", "libraries", "fingerprints")
    if os.path.exists(manifest_path):
        with io.open(manifest_path, encoding="utf-8") as handle:
            prior = json.load(handle)
        changed = [key for key in identity_keys if prior.get(key) != manifest.get(key)]
        if changed:
            raise SystemExit("拒绝续跑：实验身份变化 %s" % changed)
        manifest = prior
    else:
        temp = manifest_path + ".tmp"
        with io.open(temp, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)
        os.replace(temp, manifest_path)

    print("[%s] suite=%s cases=%d passes=%d records=%d" %
          (args.tag, args.suite, len(rows), args.passes, manifest["expected_records"]),
          flush=True)
    print("  服务=%s 模型=%s/%s focus默认=%s（A/B 仍显式覆盖）" %
          (status.get("cwd"), status.get("llm_model"), status.get("embed_model"),
           status.get("focus_floor_default")), flush=True)
    if args.check_only:
        return 0

    done, good_rows, compact = set(), [], False
    if os.path.exists(row_path):
        with io.open(row_path, encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    key = (record["pass"], norm(record["book"]), record["question"], record["arm"])
                except (ValueError, KeyError, TypeError):
                    compact = True
                    continue
                code = record.get("status")
                if (record.get("outcome") != "请求失败" and isinstance(code, int)
                        and 200 <= code < 300):
                    if key in done:
                        raise SystemExit("结果重复键，第 %d 行：%r" % (line_no, key))
                    done.add(key)
                    good_rows.append(record)
                else:
                    compact = True
    if compact:
        temp = row_path + ".resume.tmp"
        with io.open(temp, "w", encoding="utf-8") as handle:
            for record in good_rows:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        os.replace(temp, row_path)

    rng = random.Random(args.seed)
    base_order = list(rows)
    rng.shuffle(base_order)
    schedule = []
    for pass_index in range(1, args.passes + 1):
        ordered = base_order if pass_index % 2 else list(reversed(base_order))
        for case_index, row in enumerate(ordered):
            digest = hashlib.sha256((str(args.seed) + "\0" + norm(row["book"]) + "\0" +
                                     row["question"]).encode("utf-8")).digest()[0]
            first_b = bool(digest & 1)
            if pass_index % 2 == 0:
                first_b = not first_b
            arms = ("B", "A") if first_b else ("A", "B")
            for position, arm in enumerate(arms, 1):
                schedule.append((pass_index, case_index, position, row, arm))

    todo = [item for item in schedule
            if (item[0], norm(item[3]["book"]), item[3]["question"], item[4]) not in done]
    print("  已完成 %d，本次待跑 %d" % (len(done), len(todo)), flush=True)
    started_all = time.time()
    for index, (pass_index, case_index, position, row, arm) in enumerate(todo, 1):
        started = time.time()
        requested = arm == "B"
        code, payload = call(base, row, by_norm[norm(row["book"])], requested)
        outcome, answer, abstained = score(row, code, payload)
        focus = payload.get("focus_floor") or {}
        if 200 <= int(code or 0) < 300 and focus.get("enabled") is not requested:
            outcome = "请求失败"
            payload["error"] = "服务未回显实际 focus 开关：requested=%r actual=%r" % (
                requested, focus.get("enabled"))
        agent = payload.get("agent") or {}
        audit = agent.get("support_audit") or {}
        sources = []
        for source in (payload.get("sources") or [])[:12]:
            sources.append({key: source.get(key) for key in
                            ("label", "page", "doc", "library", "distance") if key in source})
        record = {
            "pass": pass_index, "case_index": case_index, "pair_position": position,
            "arm": arm, "focus_requested": requested, "hybrid_requested": False,
            "focus_enabled": focus.get("enabled"), "focus_blocked": focus.get("blocked"),
            "focus_overrode": focus.get("overrode"), "focus_phrase": focus.get("focus"),
            "question": row["question"], "book": row["book"], "type": row.get("type"),
            "expect": row.get("expect"), "keywords": row.get("keywords") or [],
            "term": row.get("term"), "status": code, "outcome": outcome,
            "abstained": abstained, "rounds": agent.get("rounds"),
            "cite_ok": bool((payload.get("cite_check") or {}).get("ok")),
            "confidence": (agent.get("confidence") or {}).get("level"),
            "pruned": audit.get("pruned"), "orphaned": audit.get("orphaned"),
            "unknown": audit.get("unknown"), "stop_reason": agent.get("stop_reason"),
            "retrieval": payload.get("retrieval"), "tokens": payload.get("tokens"),
            "elapsed": round(time.time() - started, 2), "answer": answer,
            "sources": sources, "error": payload.get("error"),
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
        }
        with io.open(row_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        if index % 20 == 0 or index == len(todo):
            elapsed = (time.time() - started_all) / 60.0
            remain = elapsed / index * (len(todo) - index) if index else 0
            print("%5d/%-5d pass=%d 已用 %.1f 分，预计剩 %.1f 分" %
                  (index, len(todo), pass_index, elapsed, remain), flush=True)
    print("[%s] 完成，用时 %.1f 分钟" %
          (args.tag, (time.time() - started_all) / 60.0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
