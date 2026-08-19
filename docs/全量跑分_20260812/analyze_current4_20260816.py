# -*- coding: utf-8 -*-
"""Validate current4 against the established same-suite WebUI floor and final3."""
from collections import Counter
import hashlib
import json
import os
import re


HERE = os.path.dirname(os.path.abspath(__file__))
CURRENT = os.path.join(HERE, "deep_current4_en_20260816_rows.jsonl")
MANIFEST = os.path.join(HERE, "deep_current4_en_20260816_manifest.json")
FINAL3 = os.path.join(HERE, "deep_final3_en_20260816_rows.jsonl")
EXPECTED_ROWS = 1007
OLD_HIT_FLOOR = 534                 # established same-suite floor, not a noisy single run
OLD_REFUSAL_FLOOR = 275
OLD_FAB_CEILING = 25


def load(path):
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def key(row):
    return str(row.get("book") or ""), str(row.get("question") or ""), str(row.get("type") or "")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def duplicate_adjacent_citations(answer):
    pair = re.compile(r"\[([^\]]+)\]\s+\[([^\]]+)\]")
    for left, right in pair.findall(str(answer or "")):
        norm = lambda text: re.sub(r"\s+", "", text).lower().replace("p", "p.", 1).replace("p..", "p.")
        if norm(left) == norm(right):
            return True
    return False


current = load(CURRENT)
baseline = load(FINAL3)
manifest = json.load(open(MANIFEST, encoding="utf-8"))
current_map = {key(row): row for row in current}
baseline_map = {key(row): row for row in baseline}
duplicates = len(current) - len(current_map)
counts = Counter(row.get("outcome") for row in current)
confidence = Counter(row.get("confidence") or "missing" for row in current)
http_errors = [row for row in current if not isinstance(row.get("status"), int)
               or not 200 <= row["status"] < 300 or row.get("outcome") == "请求失败"]
cite_failures = [row for row in current if not row.get("abstained") and not row.get("cite_ok")]
adjacent_duplicates = [row for row in current if duplicate_adjacent_citations(row.get("answer"))]
shared = sorted(set(current_map) & set(baseline_map))
migrations = Counter((baseline_map[item].get("outcome"), current_map[item].get("outcome"))
                     for item in shared if baseline_map[item].get("outcome") != current_map[item].get("outcome"))

checks = {
    "rows_1007": len(current) == EXPECTED_ROWS,
    "unique_composite_keys": duplicates == 0,
    "manifest_completed": manifest.get("completed_rows") == EXPECTED_ROWS,
    "fingerprints_stable": manifest.get("fingerprints") == manifest.get("end_fingerprints"),
    "http_errors_zero": not http_errors,
    "citation_failures_zero": not cite_failures,
    "adjacent_duplicate_citations_zero": not adjacent_duplicates,
    "hit_at_least_established_floor": counts["命中"] >= OLD_HIT_FLOOR,
    "refusal_at_least_established_floor": counts["拒答正确"] >= OLD_REFUSAL_FLOOR,
    "fabrication_at_most_established_ceiling": counts["编造"] <= OLD_FAB_CEILING,
}
report = {
    "current_path": CURRENT,
    "current_sha256": sha256(CURRENT),
    "rows": len(current),
    "unique_keys": len(current_map),
    "outcomes": dict(counts),
    "confidence": dict(confidence),
    "http_errors": len(http_errors),
    "citation_failures": len(cite_failures),
    "adjacent_duplicate_citations": len(adjacent_duplicates),
    "paired_with_final3": len(shared),
    "paired_migrations": {"%s -> %s" % pair: count for pair, count in sorted(migrations.items())},
    "runtime": (manifest.get("service_config") or {}).get("runtime"),
    "checks": checks,
    "passed": all(checks.values()),
}
print(json.dumps(report, ensure_ascii=False, indent=2))
if not report["passed"]:
    raise SystemExit(1)
