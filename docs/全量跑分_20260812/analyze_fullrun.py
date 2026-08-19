# -*- coding: utf-8 -*-
"""可复用的 WebUI 全量验收器：复合身份、指纹、库快照、引用和指标一起检查。"""
import argparse
from collections import Counter
import hashlib
import io
import json
import os
import re

from eval_compare import build_question_index, match_question_row, row_key


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EVAL = os.path.join(PROJECT_ROOT, "eval", "eval_ALL.jsonl")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path):
    with io.open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_map(rows, question_index, label):
    result = {}
    first = {}
    for line_no, row in enumerate(rows, 1):
        matched = match_question_row(row, question_index)
        key = row_key(matched)
        if key in result:
            raise ValueError("%s 第 %d 行与第 %d 行复合键重复：%r" %
                             (label, line_no, first[key], key))
        result[key] = row
        first[key] = line_no
    return result


def duplicate_adjacent_citations(answer):
    pair = re.compile(r"\[([^\]]+)\]\s+\[([^\]]+)\]")

    def normalized(value):
        text = re.sub(r"\s+", "", str(value or "")).casefold()
        return re.sub(r"^p\.?", "p.", text)

    return any(normalized(left) == normalized(right)
               for left, right in pair.findall(str(answer or "")))


def write_json(path, payload):
    temp = path + ".%d.tmp" % os.getpid()
    with io.open(temp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


parser = argparse.ArgumentParser()
parser.add_argument("tag")
parser.add_argument("--baseline-tag", default="deep_final3_en_20260816")
parser.add_argument("--expected-rows", type=int, default=1007)
parser.add_argument("--hit-floor", type=int, default=534)
parser.add_argument("--refusal-floor", type=int, default=275)
parser.add_argument("--fabrication-ceiling", type=int, default=25)
parser.add_argument("--require-library-snapshot", action="store_true")
args = parser.parse_args()

if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.tag):
    raise SystemExit("tag 非法")
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", args.baseline_tag):
    raise SystemExit("baseline tag 非法")

rows_path = os.path.join(HERE, "%s_rows.jsonl" % args.tag)
manifest_path = os.path.join(HERE, "%s_manifest.json" % args.tag)
baseline_path = os.path.join(HERE, "%s_rows.jsonl" % args.baseline_tag)
output_path = os.path.join(HERE, "%s_analysis.json" % args.tag)
for required in (rows_path, manifest_path, baseline_path, EVAL):
    if not os.path.isfile(required):
        raise SystemExit("缺少验收输入：%s" % required)

current = load_jsonl(rows_path)
baseline = load_jsonl(baseline_path)
eval_rows = load_jsonl(EVAL)
question_index = build_question_index(eval_rows)
current_map = canonical_map(current, question_index, "current")
baseline_map = canonical_map(baseline, question_index, "baseline")
manifest = json.load(io.open(manifest_path, encoding="utf-8"))

counts = Counter(row.get("outcome") for row in current)
confidence = Counter(row.get("confidence") or "missing" for row in current)
http_errors = [row for row in current if not isinstance(row.get("status"), int)
               or not 200 <= row["status"] < 300 or row.get("outcome") == "请求失败"]
cite_failures = [row for row in current if not row.get("abstained") and not row.get("cite_ok")]
adjacent_duplicates = [row for row in current
                       if duplicate_adjacent_citations(row.get("answer"))]
shared = sorted(set(current_map) & set(baseline_map))
migrations = Counter(
    (baseline_map[key].get("outcome"), current_map[key].get("outcome"))
    for key in shared if baseline_map[key].get("outcome") != current_map[key].get("outcome"))

snapshot = manifest.get("libraries") or []
end_snapshot = manifest.get("end_libraries") or []
snapshot_ids = {item.get("id") for item in snapshot if item.get("id")}
row_library_ids = {row.get("library_id") for row in current if row.get("library_id")}
snapshot_ok = (bool(snapshot) and snapshot == end_snapshot
               and (not row_library_ids or row_library_ids <= snapshot_ids))
rows_hash = sha256(rows_path)
recorded_rows_hash = manifest.get("rows_sha256")

checks = {
    "rows_expected": len(current) == args.expected_rows,
    "unique_canonical_keys": len(current_map) == len(current),
    "manifest_completed": manifest.get("completed_rows") == args.expected_rows,
    "fingerprints_stable": manifest.get("fingerprints") == manifest.get("end_fingerprints"),
    "rows_hash_matches": not recorded_rows_hash or recorded_rows_hash == rows_hash,
    "library_snapshot": snapshot_ok if args.require_library_snapshot else True,
    "http_errors_zero": not http_errors,
    "citation_failures_zero": not cite_failures,
    "adjacent_duplicate_citations_zero": not adjacent_duplicates,
    "hit_at_least_floor": counts["命中"] >= args.hit_floor,
    "refusal_at_least_floor": counts["拒答正确"] >= args.refusal_floor,
    "fabrication_at_most_ceiling": counts["编造"] <= args.fabrication_ceiling,
}
report = {
    "tag": args.tag,
    "rows_path": rows_path,
    "rows_sha256": rows_hash,
    "rows": len(current),
    "unique_canonical_keys": len(current_map),
    "outcomes": dict(counts),
    "confidence": dict(confidence),
    "http_errors": len(http_errors),
    "citation_failures": len(cite_failures),
    "adjacent_duplicate_citations": len(adjacent_duplicates),
    "library_snapshot_count": len(snapshot),
    "library_snapshot_stable": snapshot == end_snapshot,
    "row_library_ids": len(row_library_ids),
    "paired_with_baseline": len(shared),
    "paired_migrations": {"%s -> %s" % pair: count
                          for pair, count in sorted(migrations.items())},
    "runtime": (manifest.get("service_config") or {}).get("runtime"),
    "checks": checks,
    "passed": all(checks.values()),
}
write_json(output_path, report)
print(json.dumps(report, ensure_ascii=False, indent=2))
if not report["passed"]:
    raise SystemExit(1)
