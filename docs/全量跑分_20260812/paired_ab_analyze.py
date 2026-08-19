# -*- coding: utf-8 -*-
"""分析 paired_ab_run.py 结果；不把编造净额冒充“没有新增危险失败”。"""
from __future__ import print_function

import collections
import hashlib
import io
import json
import math
import os
import re
import sys
import urllib.request

from eval_compare import normalize_book

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, payload):
    temp = path + ".%d.tmp" % os.getpid()
    with io.open(temp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def stable_status(status):
    keys = ("ollama_host", "llm_model", "vl_model", "embed_model", "offline",
            "relevance_trim", "context_budget", "budget_escalated", "top_k",
            "evidence_floor", "style_gate_max", "widen_refusal", "keyword_df_ratio", "cwd")
    return {key: status.get(key) for key in keys if key in status}


def library_snapshot(items):
    keep = ("id", "name", "source", "status", "chunks", "built_at", "db_path")
    return sorted(({key: item.get(key) for key in keep if key in item} for item in items),
                  key=lambda item: str(item.get("id")))


def duplicate_adjacent_citations(answer):
    pair = re.compile(r"\[([^\]]+)\]\s+\[([^\]]+)\]")

    def normalized(value):
        text = re.sub(r"\s+", "", str(value or "")).casefold()
        return re.sub(r"^p\.?", "p.", text)

    return any(normalized(left) == normalized(right)
               for left, right in pair.findall(str(answer or "")))


def exact_mcnemar(left_only, right_only):
    """两侧精确二项检验；返回双尾 p。"""
    n = left_only + right_only
    if not n:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(left_only, right_only) + 1)) / float(2 ** n)
    return min(1.0, 2.0 * tail)


def load(path):
    rows = {}
    with io.open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (int(row["pass"]), normalize_book(row["book"]), row["question"], row["arm"])
            if key in rows:
                raise SystemExit("重复键 %s:%d %r" % (path, line_no, key))
            if row.get("outcome") == "请求失败" or not (200 <= int(row.get("status", 0)) < 300):
                raise SystemExit("结果含请求失败 %s:%d" % (path, line_no))
            rows[key] = row
    return rows


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "paired_en_20260815"
    path = tag if os.path.isabs(tag) else os.path.join(HERE, tag + "_rows.jsonl")
    manifest_path = os.path.splitext(path)[0].replace("_rows", "_manifest") + ".json"
    if not os.path.isfile(manifest_path):
        raise SystemExit("缺少实验 manifest：%s" % manifest_path)
    with io.open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    rows = load(path)
    expected_records = int(manifest.get("expected_records") or 0)
    expected_cases = int(manifest.get("expected_cases") or 0)
    expected_passes = int(manifest.get("passes") or 0)
    if len(rows) != expected_records:
        raise SystemExit("结果未完整：实际 %d / manifest 预期 %d" %
                         (len(rows), expected_records))
    for key, row in rows.items():
        expected_hybrid = key[3] == "B"
        if row.get("hybrid_requested") is not expected_hybrid:
            raise SystemExit("A/B 显式开关与 arm 不一致：%r" % (key,))
    passes = sorted({key[0] for key in rows})
    if passes != list(range(1, expected_passes + 1)):
        raise SystemExit("pass 不完整：实际 %r / 预期 1..%d" % (passes, expected_passes))

    start_fingerprints = manifest.get("fingerprints") or {}
    current_fingerprints = {}
    missing_fingerprint_files = []
    for relative in start_fingerprints:
        candidate = os.path.join(ROOT, relative)
        if not os.path.isfile(candidate):
            missing_fingerprint_files.append(relative)
            continue
        current_fingerprints[relative] = sha256(candidate)
    fingerprints_stable = (not missing_fingerprint_files
                           and current_fingerprints == start_fingerprints)
    live_error = ""
    current_service, current_libraries = {}, []
    try:
        base = "http://127.0.0.1:%s" % manifest.get("port")
        with urllib.request.urlopen(base + "/api/status", timeout=30) as response:
            current_service = stable_status(json.loads(response.read().decode("utf-8")))
        with urllib.request.urlopen(base + "/api/libraries", timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
        current_libraries = library_snapshot(
            payload.get("libraries") or payload.get("items") or payload)
    except Exception as exc:
        live_error = "%s: %s" % (type(exc).__name__, str(exc)[:200])

    cite_failures = [row for row in rows.values()
                     if not row.get("abstained") and not row.get("cite_ok")]
    adjacent_duplicates = [row for row in rows.values()
                           if duplicate_adjacent_citations(row.get("answer"))]
    summary = {
        "file": os.path.basename(path), "records": len(rows), "passes": {},
        "rows_sha256": sha256(path), "analyzer_sha256": sha256(__file__),
        "fingerprints_current": current_fingerprints,
        "missing_fingerprint_files": missing_fingerprint_files,
        "live_service": current_service, "live_library_count": len(current_libraries),
        "live_error": live_error, "citation_failures": len(cite_failures),
        "adjacent_duplicate_citations": len(adjacent_duplicates),
    }
    new_by_pass, cured_by_pass = {}, {}
    all_complete = True

    for pass_index in passes:
        a = {(book, question): row for (p, book, question, arm), row in rows.items()
             if p == pass_index and arm == "A"}
        b = {(book, question): row for (p, book, question, arm), row in rows.items()
             if p == pass_index and arm == "B"}
        if set(a) != set(b):
            raise SystemExit("pass %d A/B 键不一致：A=%d B=%d" % (pass_index, len(a), len(b)))
        keys = sorted(a)
        if len(keys) != expected_cases:
            raise SystemExit("pass %d 题数 %d / manifest 预期 %d" %
                             (pass_index, len(keys), expected_cases))
        ca = collections.Counter(a[key]["outcome"] for key in keys)
        cb = collections.Counter(b[key]["outcome"] for key in keys)
        hit_gain = sum(a[k]["outcome"] != "命中" and b[k]["outcome"] == "命中" for k in keys)
        hit_lost = sum(a[k]["outcome"] == "命中" and b[k]["outcome"] != "命中" for k in keys)
        fab_new = {k for k in keys if a[k]["outcome"] != "编造" and b[k]["outcome"] == "编造"}
        fab_cured = {k for k in keys if a[k]["outcome"] == "编造" and b[k]["outcome"] != "编造"}
        new_by_pass[pass_index], cured_by_pass[pass_index] = fab_new, fab_cured
        record = {
            "cases": len(keys), "A": dict(ca), "B": dict(cb),
            "hit_gained": hit_gain, "hit_lost": hit_lost,
            "hit_delta": cb["命中"] - ca["命中"],
            "hit_p": exact_mcnemar(hit_gain, hit_lost),
            "fab_new": len(fab_new), "fab_cured": len(fab_cured),
            "fab_delta": cb["编造"] - ca["编造"],
            "fab_p": exact_mcnemar(len(fab_new), len(fab_cured)),
        }
        summary["passes"][str(pass_index)] = record
        all_complete = all_complete and len(keys) > 0
        print("pass %d  cases=%d" % (pass_index, len(keys)))
        print("  A 命中=%d 编造=%d  B 命中=%d 编造=%d" %
              (ca["命中"], ca["编造"], cb["命中"], cb["编造"]))
        print("  命中 gained/lost=%d/%d delta=%+d McNemar p=%.6g" %
              (hit_gain, hit_lost, record["hit_delta"], record["hit_p"]))
        print("  编造 new/cured=%d/%d delta=%+d McNemar p=%.6g" %
              (len(fab_new), len(fab_cured), record["fab_delta"], record["fab_p"]))

    persistent_new = set.intersection(*(new_by_pass[p] for p in passes)) if passes else set()
    persistent_cured = set.intersection(*(cured_by_pass[p] for p in passes)) if passes else set()
    summary["persistent_new"] = [list(key) for key in sorted(persistent_new)]
    summary["persistent_cured"] = [list(key) for key in sorted(persistent_cured)]
    repeatability = {}
    for arm in ("A", "B"):
        grouped = collections.defaultdict(list)
        for (pass_index, book, question, row_arm), row in rows.items():
            if row_arm == arm:
                grouped[(book, question)].append((pass_index, row))
        outcome_flips, abstain_flips = [], []
        for key, records in grouped.items():
            ordered = [row for _pass, row in sorted(records)]
            if len({row.get("outcome") for row in ordered}) > 1:
                outcome_flips.append(list(key))
            if len({bool(row.get("abstained")) for row in ordered}) > 1:
                abstain_flips.append(list(key))
        repeatability[arm] = {
            "cases": len(grouped), "outcome_flip_count": len(outcome_flips),
            "abstain_flip_count": len(abstain_flips),
            "outcome_flips": sorted(outcome_flips), "abstain_flips": sorted(abstain_flips),
        }
    summary["repeatability"] = repeatability
    hit_ok = all(item["hit_delta"] > 0 and item["hit_p"] < 0.01
                 for item in summary["passes"].values())
    fab_counts_ok = all(item["fab_delta"] <= 0 for item in summary["passes"].values())
    safety_ok = fab_counts_ok and not persistent_new
    summary["pre_registered"] = {
        "complete": all_complete, "hit_ok": hit_ok, "fab_counts_ok": fab_counts_ok,
        "persistent_new_count": len(persistent_new), "safety_ok_before_manual_review": safety_ok,
        "candidate_for_default": bool(all_complete and hit_ok and safety_ok),
    }
    integrity = {
        "records_complete": len(rows) == expected_records,
        "fingerprints_stable": fingerprints_stable,
        "service_stable": not live_error and current_service == manifest.get("service"),
        "libraries_stable": not live_error and current_libraries == manifest.get("libraries"),
        "http_errors_zero": True,  # load() 已对任一非 2xx/请求失败 fail-fast
        "citation_failures_zero": not cite_failures,
        "adjacent_duplicate_citations_zero": not adjacent_duplicates,
    }
    summary["integrity"] = integrity
    summary["passed_integrity"] = all(integrity.values())
    print("\n持续性新增编造 %d；持续性修复编造 %d" %
          (len(persistent_new), len(persistent_cured)))
    for key in sorted(persistent_new):
        print("  NEW %s :: %s" % key)
    print("\n预注册自动部分：hit_ok=%s fab_counts_ok=%s persistent_new=%d candidate=%s" %
          (hit_ok, fab_counts_ok, len(persistent_new),
           summary["pre_registered"]["candidate_for_default"]))
    print("同臂跨遍翻转：A outcome=%d abstain=%d；B outcome=%d abstain=%d" %
          (repeatability["A"]["outcome_flip_count"], repeatability["A"]["abstain_flip_count"],
           repeatability["B"]["outcome_flip_count"], repeatability["B"]["abstain_flip_count"]))
    print("完整性：%s" % integrity)
    if not all_complete:
        print("结论：实验不完整，不得据此决定默认配置。")
    elif persistent_new:
        print("结论：存在持续性新增编造，按预注册判据不得设为全局默认；仍需人工审阅失败形态。")
    elif not fab_counts_ok:
        print("结论：至少一遍编造总数上升，按预注册判据不得设为全局默认。")
    elif not hit_ok:
        print("结论：命中收益未在每一遍同时满足方向与显著性门槛，不是默认开启候选。")
    else:
        print("结论：自动条件通过；是否默认开启仍须中文门槛与新增编造人工审阅。")
    out = os.path.splitext(path)[0] + "_analysis.json"
    write_json(out, summary)
    print("分析已写入 %s" % os.path.basename(out))
    return 0 if summary["passed_integrity"] else 1


if __name__ == "__main__":
    sys.exit(main())
