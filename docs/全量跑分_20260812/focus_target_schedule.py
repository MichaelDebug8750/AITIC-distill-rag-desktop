# -*- coding: utf-8 -*-
"""Print every pre-audit focus-floor target in the frozen A/B schedule.

The target set comes from the deterministic retrieval audit.  Keeping a second,
hand-written subset here once hid six candidates from the manual-review plan.
"""
import json
import os
import random

from paired_ab_run import load_suite, norm

HERE = os.path.dirname(__file__)
with open(os.path.join(HERE, "focus_override_20260816_analysis.json"), encoding="utf-8") as handle:
    audit = json.load(handle)
TARGET_TERMS = {
    str(item.get("dataset_term") or "").casefold()
    for item in audit.get("blocked_answerable_focus_hits", [])
    if str(item.get("dataset_term") or "").strip()
}

_paths, rows = load_suite("english")
repaired_manifest = os.path.join(HERE, "focus_floor_en_repaired_20260816_manifest.json")
manifest_path = (repaired_manifest if os.path.exists(repaired_manifest)
                 else os.path.join(HERE, "focus_floor_en_20260816_manifest.json"))
with open(manifest_path, encoding="utf-8") as handle:
    manifest = json.load(handle)
library_names = {
    norm(value) for item in manifest["libraries"] for value in (item.get("source"), item.get("name"))
    if value
}
rows = [row for row in rows if norm(row["book"]) in library_names]
rng = random.Random(20260816)
rng.shuffle(rows)
for pass_index in (1, 2):
    ordered = rows if pass_index == 1 else list(reversed(rows))
    for case_index, row in enumerate(ordered):
        if str(row.get("term") or "").casefold() in TARGET_TERMS:
            # Every case has two adjacent records; record positions are one-based.
            first_record = (pass_index - 1) * len(rows) * 2 + case_index * 2 + 1
            print("pass=%d case=%d records=%d-%d %s" %
                  (pass_index, case_index, first_record, first_record + 1, row["question"]))
