# -*- coding: utf-8 -*-
"""Run the accepted Web evaluation contract through DesktopBackend.ask_stream.

The desktop application executes the same in-process ``webui`` functions as the HTTP
application, but its production path is the SSE adapter in ``DesktopBackend``.  This
runner therefore measures that exact path, preserves full answers for manual audit,
and refuses to resume if code, cases, runtime configuration, or knowledge bases drift.

Usage::

    python packaging/desktop_full_eval.py TAG --suite en|cn [--check-only]
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import glob
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "全量跑分_20260812"
NO_REF = "[NO REFERENCE FOUND]"
CITE = re.compile(r"\[[^\]]+\]")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(RESULTS))

from desktop_app.backend import BackendError, DesktopBackend  # noqa: E402
from eval_compare import row_key  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def norm(value: Any) -> str:
    value = os.path.splitext(str(value or ""))[0]
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", value).casefold()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def load_cases(suite: str) -> tuple[list[dict[str, Any]], list[Path]]:
    if suite == "en":
        files = [ROOT / "eval" / "eval_ALL.jsonl"]
    else:
        files = [Path(value) for value in sorted(glob.glob(str(RESULTS / "eval_cn*.jsonl")))]
    rows = []
    for path in files:
        with io.open(path, encoding="utf-8-sig") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows, files


def interleave(cases: list[tuple[dict[str, Any], str, str]]) -> list[tuple[dict[str, Any], str, str]]:
    buckets: dict[str, list[tuple[dict[str, Any], str, str]]] = defaultdict(list)
    order = []
    for case in cases:
        if case[2] not in buckets:
            order.append(case[2])
        buckets[case[2]].append(case)
    result = []
    for index in range(max((len(buckets[key]) for key in order), default=0)):
        result.extend(buckets[key][index] for key in order if index < len(buckets[key]))
    return result


def library_snapshot(items: list[dict[str, Any]], used: set[str]) -> list[dict[str, Any]]:
    keys = ("id", "name", "source", "chunks", "built_at", "status")
    return sorted(
        [{key: item.get(key) for key in keys} for item in items if item.get("id") in used],
        key=lambda item: str(item.get("id") or ""),
    )


def classify(case: dict[str, Any], payload: dict[str, Any]) -> tuple[str, bool | None]:
    answer = str(payload.get("answer") or "").strip()
    abstained = bool(payload.get("abstained"))
    if case.get("expect") == "abstain" or case.get("type") == "unanswerable":
        return ("拒答正确" if abstained and answer == NO_REF else "编造"), None
    if abstained:
        return "过度拒答", None
    keywords = [str(value).casefold() for value in (case.get("keywords") or [])]
    if not keywords:
        return "未判定", None
    body = CITE.sub("", answer).casefold()
    hit = any(value in body for value in keywords)
    return ("命中" if hit else "未命中"), hit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("--suite", choices=("en", "cn"), required=True)
    parser.add_argument("--project-root", default=str(ROOT))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.tag):
        raise SystemExit("tag 只能含字母、数字、下划线和连字符")

    rows_path = RESULTS / (args.tag + "_rows.jsonl")
    manifest_path = RESULTS / (args.tag + "_manifest.json")
    backend = DesktopBackend(args.project_root)
    try:
        status = backend.status()
        all_libraries = list((backend.libraries() or {}).get("libraries") or [])
        ready = [item for item in all_libraries if str(item.get("status") or "ready") == "ready"]
        by_norm: dict[str, str] = {}
        for item in ready:
            for value in (item.get("source"), item.get("name")):
                if value:
                    by_norm.setdefault(norm(value), str(item.get("id")))

        source_rows, case_files = load_cases(args.suite)
        cases = []
        missing = set()
        seen = set()
        for row in source_rows:
            library_id = by_norm.get(norm(row.get("book")))
            if not library_id:
                missing.add(str(row.get("book") or ""))
                continue
            key = row_key(row)
            if key in seen:
                raise RuntimeError("题集复合键重复：%r" % (key,))
            seen.add(key)
            cases.append((row, library_id, os.path.splitext(str(row.get("book") or ""))[0]))

        expected = 1007 if args.suite == "en" else 104
        if len(cases) != expected:
            raise RuntimeError(
                "%s 桌面全量应匹配 %d 题，实际 %d；缺库示例：%s"
                % (args.suite, expected, len(cases), "、".join(sorted(missing)[:8])))
        used = {case[1] for case in cases}
        snapshot = library_snapshot(ready, used)
        if len(snapshot) != len(used):
            raise RuntimeError("无法为全部知识库建立快照")

        fingerprint_paths = case_files + [
            ROOT / "code" / "webui.py",
            ROOT / "code" / "test_pipeline.py",
            ROOT / "code" / "main.py",
            ROOT / "desktop_app" / "backend.py",
            ROOT / "desktop_app" / "main_window.py",
            Path(__file__).resolve(),
        ]
        fingerprints = {str(path.relative_to(ROOT)): sha256(path) for path in fingerprint_paths}
        service_config = {key: status.get(key) for key in (
            "llm_model", "embed_model", "hybrid_default", "evidence_floor",
            "style_gate_max", "model_seed", "widen_refusal", "keyword_df_ratio",
            "runtime")}
        identity = {
            "schema": 1,
            "tag": args.tag,
            "suite": args.suite,
            "adapter": "DesktopBackend.ask_stream",
            "hybrid_request": False,
            "project_root": str(Path(args.project_root).resolve()),
            "service_config": service_config,
            "fingerprints": fingerprints,
            "libraries": snapshot,
            "expected_rows": expected,
        }
        if manifest_path.exists():
            prior = json.loads(manifest_path.read_text(encoding="utf-8"))
            for key in ("suite", "adapter", "hybrid_request", "project_root",
                        "service_config", "fingerprints", "libraries", "expected_rows"):
                if prior.get(key) != identity.get(key):
                    raise RuntimeError("现有 manifest 的 %s 已漂移，拒绝混合运行" % key)
            manifest = prior
        else:
            manifest = dict(identity, started_at=time.strftime("%Y-%m-%d %H:%M:%S"))
            atomic_json(manifest_path, manifest)

        print("[%s] suite=%s，%d 本库，%d 题，adapter=%s" %
              (args.tag, args.suite, len(snapshot), len(cases), identity["adapter"]), flush=True)
        if args.check_only:
            return 0

        done = set()
        valid_rows = []
        compact = False
        if rows_path.exists():
            with rows_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        saved = json.loads(line)
                        key = row_key(saved)
                    except (ValueError, TypeError, KeyError):
                        compact = True
                        continue
                    if saved.get("outcome") != "请求失败" and saved.get("status") == 200:
                        if key in done:
                            raise RuntimeError("结果文件复合键重复：%r" % (key,))
                        done.add(key)
                        valid_rows.append(saved)
                    else:
                        compact = True
        if compact:
            temporary = rows_path.with_suffix(".resume.tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                for saved in valid_rows:
                    handle.write(json.dumps(saved, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(rows_path)

        abstain = [case for case in cases if case[0].get("expect") == "abstain"
                   or case[0].get("type") == "unanswerable"]
        answerable = [case for case in cases if case not in abstain]
        ordered = abstain + interleave(answerable)
        todo = [case for case in ordered
                if row_key({"book": case[2], "question": case[0]["question"]}) not in done]
        print("[%s] 已完成 %d，本次待跑 %d" % (args.tag, len(done), len(todo)), flush=True)

        started = time.time()
        for index, (case, library_id, book) in enumerate(todo, 1):
            payload: dict[str, Any] = {}
            error = ""
            request_started = time.time()
            for attempt in range(3):
                try:
                    payload = backend.ask_stream(
                        str(case["question"]), libraries=[library_id], history=[],
                        mode="auto", style="standard", hybrid=False, extend=False)
                    error = ""
                    break
                except BackendError as exc:
                    error = "%s: %s" % (type(exc).__name__, str(exc)[:500])
                    if attempt < 2:
                        time.sleep(min(2 ** attempt, 4))
            if error:
                outcome, hit, status_code = "请求失败", None, 0
            else:
                outcome, hit = classify(case, payload)
                status_code = 200
            answer = str(payload.get("answer") or "").strip()
            agent = payload.get("agent") or {}
            audit = agent.get("support_audit") or {}
            record = {
                "question": case["question"], "book": book, "library_id": library_id,
                "type": case.get("type"), "expect": case.get("expect"),
                "keywords": case.get("keywords") or [], "status": status_code,
                "outcome": outcome, "hit": hit, "abstained": bool(payload.get("abstained")),
                "rounds": agent.get("rounds"),
                "cite_ok": bool((payload.get("cite_check") or {}).get("ok")),
                "confidence": (agent.get("confidence") or {}).get("level"),
                "pruned": audit.get("pruned"), "reassembly_pruned": audit.get("reassembly_pruned"),
                "orphaned": audit.get("orphaned"), "unknown": audit.get("unknown"),
                "stop_reason": agent.get("stop_reason"),
                "elapsed": round(time.time() - request_started, 1),
                "tokens": payload.get("tokens"), "answer": answer, "error": error,
            }
            with rows_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if index % 20 == 0 or index == len(todo):
                minutes = (time.time() - started) / 60
                remaining = minutes / index * (len(todo) - index) if index else 0
                print("%5d/%-5d 已用 %.1f 分钟，预计还需 %.1f 分钟" %
                      (index, len(todo), minutes, remaining), flush=True)

        end_status = backend.status()
        end_libraries = list((backend.libraries() or {}).get("libraries") or [])
        end_snapshot = library_snapshot(
            [item for item in end_libraries if str(item.get("status") or "ready") == "ready"], used)
        end_fingerprints = {str(path.relative_to(ROOT)): sha256(path) for path in fingerprint_paths}
        if end_fingerprints != fingerprints:
            raise RuntimeError("运行期间源码或题集发生变化")
        if end_snapshot != snapshot:
            raise RuntimeError("运行期间知识库快照发生变化")
        end_config = {key: end_status.get(key) for key in service_config}
        if end_config != service_config:
            raise RuntimeError("运行期间服务配置发生变化")
        with rows_path.open(encoding="utf-8") as handle:
            completed = sum(1 for line in handle if line.strip())
        manifest.update({
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "completed_rows": completed,
            "rows_sha256": sha256(rows_path),
            "end_fingerprints": end_fingerprints,
            "end_libraries": end_snapshot,
            "end_service_config": end_config,
        })
        atomic_json(manifest_path, manifest)
        print("[%s] 完成 %d/%d，用时 %.1f 分钟" %
              (args.tag, completed, expected, (time.time() - started) / 60), flush=True)
        return 0
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
