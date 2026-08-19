# -*- coding: utf-8 -*-
"""Integrity and metric gate for ``desktop_full_eval.py`` results."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "全量跑分_20260812"
sys.path.insert(0, str(RESULTS))
from eval_compare import row_key  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rows(path: Path) -> list[dict]:
    with io.open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tag")
    parser.add_argument("--suite", choices=("en", "cn"), required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.tag):
        raise SystemExit("tag 非法")
    rows_path = RESULTS / (args.tag + "_rows.jsonl")
    manifest_path = RESULTS / (args.tag + "_manifest.json")
    if not rows_path.is_file() or not manifest_path.is_file():
        raise SystemExit("缺少结果或 manifest")
    rows = load_rows(rows_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = 1007 if args.suite == "en" else 104
    keys = [row_key(row) for row in rows]
    counts = Counter(row.get("outcome") for row in rows)
    failures = [row for row in rows if row.get("status") != 200 or row.get("outcome") == "请求失败"]
    cite_failures = [row for row in rows if not row.get("abstained") and not row.get("cite_ok")]
    if args.suite == "en":
        metric_checks = {
            "hit_at_least_web_current5": counts["命中"] >= 548,
            "refusal_at_least_web_current5": counts["拒答正确"] >= 277,
            "fabrication_at_most_web_current5": counts["编造"] <= 23,
            "project_report_hit_floor": counts["命中"] >= 534,
            "project_report_refusal_floor": counts["拒答正确"] >= 275,
            "project_report_fabrication_ceiling": counts["编造"] <= 25,
        }
    else:
        metric_checks = {
            "hit_at_least_web_floor": counts["命中"] >= 59,
            "refusal_equals_web": counts["拒答正确"] == 40,
            "fabrication_equals_web": counts["编造"] == 0,
        }
    checks = {
        "rows_expected": len(rows) == expected,
        "unique_keys": len(set(keys)) == len(keys),
        "manifest_complete": manifest.get("completed_rows") == expected,
        "rows_hash": manifest.get("rows_sha256") == sha256(rows_path),
        "fingerprints_stable": manifest.get("fingerprints") == manifest.get("end_fingerprints"),
        "libraries_stable": manifest.get("libraries") == manifest.get("end_libraries"),
        "service_stable": manifest.get("service_config") == manifest.get("end_service_config"),
        "desktop_adapter": manifest.get("adapter") == "DesktopBackend.ask_stream",
        "http_errors_zero": not failures,
        "citation_failures_zero": not cite_failures,
        **metric_checks,
    }
    report = {
        "tag": args.tag,
        "suite": args.suite,
        "rows": len(rows),
        "outcomes": dict(counts),
        "confidence": dict(Counter(row.get("confidence") or "missing" for row in rows)),
        "http_errors": len(failures),
        "citation_failures": len(cite_failures),
        "reassembly_pruned_rows": sum(bool(row.get("reassembly_pruned")) for row in rows),
        "rows_sha256": sha256(rows_path),
        "checks": checks,
        "passed": all(checks.values()),
    }
    output = RESULTS / (args.tag + "_analysis.json")
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
