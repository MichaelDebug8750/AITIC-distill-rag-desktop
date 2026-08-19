# -*- coding: utf-8 -*-
"""Print paired outcome changes between two JSONL evaluation runs."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def key(row: dict) -> tuple[str, str]:
    return str(row.get("book", "")), str(row.get("question", ""))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--old-pass", type=int)
    parser.add_argument("--old-arm")
    args = parser.parse_args()

    old_rows = read_rows(args.old)
    if args.old_pass is not None:
        old_rows = [row for row in old_rows if row.get("pass") == args.old_pass]
    if args.old_arm:
        old_rows = [row for row in old_rows if row.get("arm") == args.old_arm]
    new_rows = read_rows(args.new)
    old_by_key = {key(row): row for row in old_rows}
    changes: list[dict] = []
    for new in new_rows:
        old = old_by_key.get(key(new))
        if old is None or old.get("outcome") == new.get("outcome"):
            continue
        changes.append(
            {
                "book": new.get("book"),
                "question": new.get("question"),
                "old": old.get("outcome"),
                "new": new.get("outcome"),
                "new_pruned": new.get("pruned"),
                "new_reassembly_pruned": new.get("reassembly_pruned"),
                "new_stop_reason": new.get("stop_reason"),
            }
        )
    summary = Counter((item["old"], item["new"]) for item in changes)
    print(
        json.dumps(
            {
                "old_rows": len(old_rows),
                "new_rows": len(new_rows),
                "paired": len(set(old_by_key).intersection(map(key, new_rows))),
                "change_count": len(changes),
                "summary": {f"{old} -> {new}": count for (old, new), count in summary.items()},
                "changes": changes,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
