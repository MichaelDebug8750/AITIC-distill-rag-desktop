# -*- coding: utf-8 -*-
"""同一服务上的逐题 A/B 配对跑分；显式传 hybrid，不依赖服务默认配置。

用法：
  paired_ab_run.py <port> <tag> --suite english|chinese [--passes 2] [--seed 20260815]
  paired_ab_run.py <port> <tag> --suite english --check-only

结果可断点恢复。身份为 (pass, book, question, arm)，恢复时会核对 manifest 中的服务、
知识库、题集、脚本与核心源码指纹；任一变化都拒绝把不同构建混进同一结果。
"""
from __future__ import print_function

import argparse
import hashlib
import io
import json
import os
import platform
import random
import re
import sys
import time
import urllib.error
import urllib.request

from eval_compare import normalize_book

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CODE = os.path.join(ROOT, "code")
NO_REF = "[NO REFERENCE FOUND]"
CITE = re.compile(r"\[[^\]]+\]")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    temp = path + ".%d.tmp" % os.getpid()
    with io.open(temp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def fetch_json(url, timeout=180):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def norm(name):
    return normalize_book(name)


def load_suite(name):
    if name == "english":
        paths = [os.path.join(ROOT, "eval", "eval_ALL.jsonl")]
    else:
        paths = [os.path.join(HERE, item) for item in
                 ("eval_cn.jsonl", "eval_cn2.jsonl", "eval_cn_gnu_make.jsonl",
                  "eval_cn_aigc.jsonl")]
    rows = []
    for path in paths:
        if not os.path.isfile(path):
            raise SystemExit("题集不存在：%s" % path)
        with io.open(path, encoding="utf-8-sig") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    seen = set()
    for row in rows:
        key = (norm(row.get("book")), str(row.get("question") or "").strip())
        if not all(key):
            raise SystemExit("题集缺 book/question：%r" % row)
        if key in seen:
            raise SystemExit("题集复合键重复：%r" % (key,))
        seen.add(key)
    return paths, rows


def is_unanswerable(row):
    return row.get("expect") == "abstain" or row.get("type") == "unanswerable"


def score(row, status, payload):
    answer = str(payload.get("answer") or "").strip()
    abstained = bool(payload.get("abstained"))
    if not isinstance(status, int) or not (200 <= status < 300):
        return "请求失败", answer, abstained
    if is_unanswerable(row):
        return ("拒答正确" if abstained and answer == NO_REF else "编造"), answer, abstained
    if abstained:
        return "过度拒答", answer, abstained
    keywords = [str(item).casefold() for item in (row.get("keywords") or [])]
    if not keywords:
        return "未判定", answer, abstained
    body = CITE.sub("", answer).casefold()
    return ("命中" if any(item in body for item in keywords) else "未命中"), answer, abstained


def call(base, row, library_id, hybrid, attempts=3):
    body = {"question": row["question"], "libraries": [library_id], "mode": "auto",
            "style": "standard", "extend": False, "hybrid": bool(hybrid), "history": []}
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


def source_fingerprints(paths):
    targets = list(paths) + [
        os.path.join(CODE, "webui.py"), os.path.join(CODE, "webui_index.html"),
        os.path.join(CODE, "test_pipeline.py"), os.path.join(CODE, "main.py"), __file__,
    ]
    return {os.path.relpath(path, ROOT): sha256(path) for path in targets}


def stable_status(status):
    keys = ("ollama_host", "llm_model", "vl_model", "embed_model", "offline",
            "relevance_trim", "context_budget", "budget_escalated", "top_k",
            "evidence_floor", "style_gate_max", "widen_refusal", "keyword_df_ratio", "cwd")
    return {key: status.get(key) for key in keys if key in status}


def library_snapshot(items):
    keep = ("id", "name", "source", "status", "chunks", "built_at", "db_path")
    return sorted(({key: item.get(key) for key in keep if key in item} for item in items),
                  key=lambda item: str(item.get("id")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("port")
    parser.add_argument("tag")
    parser.add_argument("--suite", choices=("english", "chinese"), required=True)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.passes <= 4:
        raise SystemExit("passes 必须为 1..4")

    base = "http://127.0.0.1:%s" % args.port
    paths, rows = load_suite(args.suite)
    status = fetch_json(base + "/api/status")
    if not status.get("ready"):
        raise SystemExit("服务未 ready：%r" % status)
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
        "schema": 1, "tag": args.tag, "suite": args.suite, "port": args.port,
        "passes": args.passes, "seed": args.seed, "expected_cases": len(rows),
        "expected_records": len(rows) * args.passes * 2,
        "request_arms": {"A": False, "B": True},
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "python": platform.python_version(), "service": stable_status(status),
        "libraries": library_snapshot(libraries), "fingerprints": source_fingerprints(paths),
    }
    identity_keys = ("schema", "suite", "passes", "seed", "expected_cases",
                     "expected_records", "request_arms", "service", "libraries", "fingerprints")
    if os.path.exists(manifest_path):
        with io.open(manifest_path, encoding="utf-8") as handle:
            prior = json.load(handle)
        changed = [key for key in identity_keys if prior.get(key) != manifest.get(key)]
        if changed:
            raise SystemExit("拒绝续跑：实验身份变化 %s" % changed)
        manifest = prior
    else:
        write_json(manifest_path, manifest)

    print("[%s] suite=%s cases=%d passes=%d records=%d" %
          (args.tag, args.suite, len(rows), args.passes, manifest["expected_records"]), flush=True)
    print("  服务=%s 模型=%s/%s 默认hybrid=%s（请求仍显式覆盖）" %
          (status.get("cwd"), status.get("llm_model"), status.get("embed_model"),
           status.get("hybrid_default")), flush=True)
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
                status_code = record.get("status")
                if (record.get("outcome") != "请求失败" and isinstance(status_code, int)
                        and 200 <= status_code < 300):
                    if key in done:
                        raise SystemExit("结果重复键，第 %d 行：%r" % (line_no, key))
                    done.add(key); good_rows.append(record)
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
        code, payload = call(base, row, by_norm[norm(row["book"])], arm == "B")
        outcome, answer, abstained = score(row, code, payload)
        agent = payload.get("agent") or {}
        audit = agent.get("support_audit") or {}
        sources = []
        for source in (payload.get("sources") or [])[:12]:
            sources.append({key: source.get(key) for key in
                            ("label", "page", "doc", "library", "distance") if key in source})
        record = {
            "pass": pass_index, "case_index": case_index, "pair_position": position,
            "arm": arm, "hybrid_requested": arm == "B", "question": row["question"],
            "book": row["book"], "type": row.get("type"), "expect": row.get("expect"),
            "keywords": row.get("keywords") or [], "term": row.get("term"),
            "status": code, "outcome": outcome, "abstained": abstained,
            "rounds": agent.get("rounds"), "cite_ok": bool((payload.get("cite_check") or {}).get("ok")),
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

    # 完成声明必须由落盘结果和结束时身份共同决定。仅仅循环跑到末尾不够：
    # 最后一条请求失败、源码在中途被编辑、服务重启到另一配置或库被替换，都会让
    # “416/416”成为不可比较的假完成。
    final_keys = set()
    with io.open(row_path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                key = (int(record["pass"]), norm(record["book"]),
                       record["question"], record["arm"])
            except (ValueError, KeyError, TypeError) as exc:
                raise SystemExit("完成结果第 %d 行损坏：%s" % (line_no, type(exc).__name__))
            status_code = record.get("status")
            if (record.get("outcome") == "请求失败" or not isinstance(status_code, int)
                    or not 200 <= status_code < 300):
                raise SystemExit("完成结果仍含请求失败，第 %d 行" % line_no)
            if key in final_keys:
                raise SystemExit("完成结果复合键重复，第 %d 行：%r" % (line_no, key))
            final_keys.add(key)
    expected_keys = {(item[0], norm(item[3]["book"]), item[3]["question"], item[4])
                     for item in schedule}
    if final_keys != expected_keys:
        raise SystemExit("完成结果键不完整：实际 %d / 预期 %d" %
                         (len(final_keys), len(expected_keys)))

    end_status = stable_status(fetch_json(base + "/api/status"))
    end_payload = fetch_json(base + "/api/libraries")
    end_libraries = library_snapshot(
        end_payload.get("libraries") or end_payload.get("items") or end_payload)
    end_fingerprints = source_fingerprints(paths)
    changed = []
    if end_status != manifest.get("service"):
        changed.append("service")
    if end_libraries != manifest.get("libraries"):
        changed.append("libraries")
    if end_fingerprints != manifest.get("fingerprints"):
        changed.append("fingerprints")
    if changed:
        raise SystemExit("运行期间实验身份变化，拒绝标记完成：%s" % changed)
    manifest.update({
        "completed_records": len(final_keys),
        "completed_at": time.strftime("%Y-%m-%d %H:%M:%S%z"),
        "rows_sha256": sha256(row_path),
        "end_service": end_status,
        "end_libraries": end_libraries,
        "end_fingerprints": end_fingerprints,
    })
    write_json(manifest_path, manifest)
    print("[%s] 完成，用时 %.1f 分钟" % (args.tag, (time.time() - started_all) / 60.0), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
