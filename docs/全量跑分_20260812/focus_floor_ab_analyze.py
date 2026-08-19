# -*- coding: utf-8 -*-
"""分析焦点短语证据下限配对实验，并逐项执行预注册门槛。"""
from __future__ import print_function

import collections
import io
import json
import math
import os
import sys

from eval_compare import normalize_book
from paired_ab_run import sha256

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def exact_mcnemar(left_only, right_only):
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
                raise SystemExit("结果含请求失败 %s:%d：%s" %
                                 (path, line_no, row.get("error")))
            rows[key] = row
    return rows


def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "focus_floor_en_20260816"
    path = tag if os.path.isabs(tag) else os.path.join(HERE, tag + "_rows.jsonl")
    manifest_path = os.path.splitext(path)[0].replace("_rows", "_manifest") + ".json"
    if not os.path.isfile(manifest_path):
        raise SystemExit("缺少实验 manifest：%s" % manifest_path)
    with io.open(manifest_path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("experiment") != "focus_floor":
        raise SystemExit("manifest 不是 focus_floor 实验")

    drift = []
    for rel, expected in (manifest.get("fingerprints") or {}).items():
        target = os.path.join(ROOT, rel)
        actual = sha256(target) if os.path.isfile(target) else None
        if actual != expected:
            drift.append(rel)
    if drift:
        raise SystemExit("实验期间源码/题集指纹漂移：%s" % drift)

    rows = load(path)
    expected_records = int(manifest.get("expected_records") or 0)
    expected_cases = int(manifest.get("expected_cases") or 0)
    expected_passes = int(manifest.get("passes") or 0)
    if len(rows) != expected_records:
        raise SystemExit("结果未完整：实际 %d / manifest 预期 %d" %
                         (len(rows), expected_records))
    for key, row in rows.items():
        expected = key[3] == "B"
        if row.get("focus_requested") is not expected or row.get("focus_enabled") is not expected:
            raise SystemExit("A/B 请求/实际开关与 arm 不一致：%r" % (key,))
        if row.get("hybrid_requested") is not False or row.get("retrieval") != "vector":
            raise SystemExit("实验意外启用了非向量检索：%r" % (key,))
        if key[3] == "A" and row.get("focus_overrode"):
            raise SystemExit("A 臂不应触发例外：%r" % (key,))
        if row.get("focus_overrode"):
            if not row.get("focus_blocked") or not str(row.get("focus_phrase") or "").strip():
                raise SystemExit("例外触发记录不可解释：%r" % (key,))

    passes = sorted({key[0] for key in rows})
    if passes != list(range(1, expected_passes + 1)):
        raise SystemExit("pass 不完整：实际 %r / 预期 1..%d" % (passes, expected_passes))

    suite = manifest.get("suite")
    summary = {"file": os.path.basename(path), "suite": suite, "records": len(rows),
               "fingerprint_drift": [], "passes": {}}
    new_fab_by_pass, lost_hit_by_pass = {}, {}
    automatic_ok = True

    for pass_index in passes:
        a = {(book, question): row for (p, book, question, arm), row in rows.items()
             if p == pass_index and arm == "A"}
        b = {(book, question): row for (p, book, question, arm), row in rows.items()
             if p == pass_index and arm == "B"}
        if set(a) != set(b) or len(a) != expected_cases:
            raise SystemExit("pass %d A/B 不完整：A=%d B=%d expected=%d" %
                             (pass_index, len(a), len(b), expected_cases))
        keys = sorted(a)
        ca = collections.Counter(a[key]["outcome"] for key in keys)
        cb = collections.Counter(b[key]["outcome"] for key in keys)
        gained = {k for k in keys if a[k]["outcome"] != "命中" and b[k]["outcome"] == "命中"}
        lost = {k for k in keys if a[k]["outcome"] == "命中" and b[k]["outcome"] != "命中"}
        new_fab = {k for k in keys if a[k]["outcome"] != "编造" and b[k]["outcome"] == "编造"}
        cured_fab = {k for k in keys if a[k]["outcome"] == "编造" and b[k]["outcome"] != "编造"}
        new_fab_by_pass[pass_index], lost_hit_by_pass[pass_index] = new_fab, lost
        triggers = sum(bool(b[k].get("focus_overrode")) for k in keys)
        hit_p = exact_mcnemar(len(gained), len(lost))
        record = {
            "cases": len(keys), "A": dict(ca), "B": dict(cb), "triggered": triggers,
            "hit_gained": len(gained), "hit_lost": len(lost),
            "hit_delta": cb["命中"] - ca["命中"], "hit_p": hit_p,
            "fab_new": len(new_fab), "fab_cured": len(cured_fab),
            "fab_delta": cb["编造"] - ca["编造"],
            "fab_p": exact_mcnemar(len(new_fab), len(cured_fab)),
        }
        if suite == "english":
            floor_ok = (cb["命中"] >= 534 and cb["拒答正确"] >= 275 and cb["编造"] <= 25)
            pass_ok = (record["hit_delta"] > 0 and hit_p < 0.05 and
                       cb["编造"] <= ca["编造"] and floor_ok)
        else:
            floor_ok = (cb["命中"] >= 59 and cb["拒答正确"] == 40 and cb["编造"] == 0)
            pass_ok = floor_ok and cb["编造"] <= ca["编造"]
        record["baseline_floor_ok"] = floor_ok
        record["automatic_pass_ok"] = pass_ok
        automatic_ok = automatic_ok and pass_ok
        summary["passes"][str(pass_index)] = record
        print("pass %d cases=%d triggers=%d" % (pass_index, len(keys), triggers))
        print("  A 命中=%d 拒答正确=%d 编造=%d" %
              (ca["命中"], ca["拒答正确"], ca["编造"]))
        print("  B 命中=%d 拒答正确=%d 编造=%d" %
              (cb["命中"], cb["拒答正确"], cb["编造"]))
        print("  命中 gained/lost=%d/%d delta=%+d McNemar p=%.6g" %
              (len(gained), len(lost), record["hit_delta"], hit_p))
        print("  编造 new/cured=%d/%d delta=%+d" %
              (len(new_fab), len(cured_fab), record["fab_delta"]))

    persistent_new_fab = (set.intersection(*(new_fab_by_pass[p] for p in passes))
                          if passes else set())
    persistent_lost_hit = (set.intersection(*(lost_hit_by_pass[p] for p in passes))
                           if passes else set())
    safety_ok = not persistent_new_fab and not persistent_lost_hit
    candidate = automatic_ok and safety_ok
    summary["persistent_new_fabrication"] = [list(k) for k in sorted(persistent_new_fab)]
    summary["persistent_hit_loss"] = [list(k) for k in sorted(persistent_lost_hit)]
    summary["pre_registered"] = {
        "automatic_passes_ok": automatic_ok,
        "persistent_new_fabrication_count": len(persistent_new_fab),
        "persistent_hit_loss_count": len(persistent_lost_hit),
        "safety_ok_before_single_pass_manual_review": safety_ok,
        "candidate_for_adoption": candidate,
    }
    print("\n持续性新增编造=%d；持续性命中损失=%d；candidate=%s" %
          (len(persistent_new_fab), len(persistent_lost_hit), candidate))
    for key in sorted(persistent_new_fab):
        print("  NEW FAB %s :: %s" % key)
    for key in sorted(persistent_lost_hit):
        print("  LOST HIT %s :: %s" % key)
    print("单遍新增编造或命中损失仍须逐条人工审阅；净额不能替代失败形态审计。")

    out = os.path.splitext(path)[0] + "_analysis.json"
    with io.open(out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print("分析已写入 %s" % os.path.basename(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
