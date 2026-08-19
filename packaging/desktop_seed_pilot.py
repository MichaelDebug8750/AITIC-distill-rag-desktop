# -*- coding: utf-8 -*-
"""Evaluate one explicit seed, or the production unseeded default, on selected rows."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packaging"))


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def row_key(row: dict) -> tuple[str, str]:
    return str(row.get("book") or ""), str(row.get("question") or "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    seed_group = parser.add_mutually_exclusive_group(required=True)
    seed_group.add_argument("--seed", type=int)
    seed_group.add_argument("--unseeded", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--question", action="append", default=[],
                        help="Run the exact question even when the two source runs did not flip")
    parser.add_argument("--select-new-outcome", action="append", default=[],
                        help="Run every paired row whose new-run outcome matches this value")
    parser.add_argument("--select-type", action="append", default=[],
                        help="Limit selected rows to these evaluation case types")
    args = parser.parse_args()

    # webui reads the optional experiment seed while importing, so set or clear
    # it before DesktopBackend.  Clearing it exercises the production default.
    if args.unseeded:
        os.environ.pop("DISTILL_MODEL_SEED", None)
    else:
        os.environ["DISTILL_MODEL_SEED"] = str(args.seed)
    from desktop_app.backend import DesktopBackend
    from desktop_full_eval import classify, norm

    old = {row_key(row): row for row in read_rows(args.old)}
    new = {row_key(row): row for row in read_rows(args.new)}
    shared = old.keys() & new.keys()
    if args.question:
        wanted = {str(value) for value in args.question}
        keys = sorted(key for key in shared if key[1] in wanted)
        missing = wanted - {key[1] for key in keys}
        if missing:
            raise RuntimeError("Questions missing from paired inputs: %s" % sorted(missing))
    elif args.select_new_outcome or args.select_type:
        wanted_outcomes = {str(value) for value in args.select_new_outcome}
        wanted_types = {str(value) for value in args.select_type}
        keys = sorted(
            key for key in shared
            if (not wanted_outcomes or str(new[key].get("outcome")) in wanted_outcomes)
            and (not wanted_types or str(new[key].get("type")) in wanted_types)
        )
    else:
        keys = sorted(key for key in shared
                      if old[key].get("outcome") != new[key].get("outcome"))
    if args.limit > 0:
        # Always retain safety flips, then fill with answerable flips.
        safety = [key for key in keys if old[key].get("type") == "unanswerable"]
        answerable = [key for key in keys if key not in safety]
        keys = (safety + answerable)[:args.limit]

    backend = DesktopBackend(ROOT)
    try:
        libraries = list((backend.libraries() or {}).get("libraries") or [])
        by_norm = {}
        for item in libraries:
            if str(item.get("status") or "ready") != "ready":
                continue
            for value in (item.get("source"), item.get("name")):
                if value:
                    by_norm.setdefault(norm(value), str(item.get("id")))

        results = []
        started = time.time()
        for index, key in enumerate(keys, 1):
            # The historical Web rows predate the self-contained desktop row
            # format and may not carry ``keywords``.  Use the newer row as the
            # canonical case metadata; the old row is only the comparison arm.
            case = new[key]
            library_id = by_norm.get(norm(case.get("book")))
            if not library_id:
                raise RuntimeError("Missing library for %s" % (case.get("book"),))
            payload = backend.ask_stream(
                str(case.get("question") or ""), libraries=[library_id], history=[],
                mode="auto", style="standard", hybrid=False, extend=False)
            outcome, hit = classify(case, payload)
            results.append({
                "book": key[0], "question": key[1], "type": case.get("type"),
                "old": old[key].get("outcome"), "new": new[key].get("outcome"),
                "observed": outcome, "hit": hit, "abstained": payload.get("abstained"),
                "answer": payload.get("answer"),
            })
            if index % 20 == 0 or index == len(keys):
                print("%d/%d elapsed %.1f min" %
                      (index, len(keys), (time.time() - started) / 60), flush=True)
    finally:
        backend.close()

    counts = Counter(row["observed"] for row in results)
    report = {
        "seed": None if args.unseeded else args.seed,
        "mode": "unseeded" if args.unseeded else "explicit_seed",
        "rows": len(results),
        "outcomes": dict(counts),
        "same_as_old": sum(row["observed"] == row["old"] for row in results),
        "same_as_new": sum(row["observed"] == row["new"] for row in results),
        "answerable_hits": sum(row["observed"] == "命中" for row in results),
        "correct_refusals": sum(row["observed"] == "拒答正确" for row in results),
        "fabrications": sum(row["observed"] == "编造" for row in results),
        "elapsed_seconds": round(time.time() - started, 2),
        "results": results,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
