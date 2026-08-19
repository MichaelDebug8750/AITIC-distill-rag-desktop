# -*- coding: utf-8 -*-
"""Inspect retrieval evidence for the current Chinese desktop-eval failures.

This is deliberately retrieval-only: it does not sample the answer model and therefore
separates recall failures from generation, adoption, and semantic-pruning failures.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "全量跑分_20260812"
ROWS = RESULTS / "desktop_seed42_full_cn_20260818_rows.jsonl"
sys.path.insert(0, str(ROOT))

from desktop_app.backend import DesktopBackend  # noqa: E402


def norm(value: object) -> str:
    return re.sub(r"\s+", "", str(value or "")).casefold()


def anchors(question: str) -> list[str]:
    values = re.findall(r"[“「『\"]([^”」』\"]{2,32})[”」』\"]", question)
    values += re.findall(r"\b[A-Za-z][A-Za-z0-9_-]{3,}\b", question)
    values += re.findall(r"\b[A-Za-z]-[A-Za-z]\s+[A-Za-z]\b", question)
    return list(dict.fromkeys(value.strip() for value in values if value.strip()))


def main() -> int:
    failed = []
    with ROWS.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("outcome") in ("过度拒答", "未命中"):
                failed.append(row)

    backend = DesktopBackend(ROOT)
    try:
        reports = []
        corpus_cache = {}
        for row in failed:
            library_id = str(row["library_id"])
            if library_id not in corpus_cache:
                target = backend.webui._library_targets([library_id])[0]
                collection = backend.webui.M.chromadb.PersistentClient(
                    path=str(target["path"])
                ).get_collection(backend.webui.M.COLLECTION)
                raw = collection.get(include=["documents", "metadatas"])
                corpus_cache[library_id] = list(zip(
                    raw.get("documents") or [], raw.get("metadatas") or []
                ))
            corpus = corpus_cache[library_id]
            corpus_hits = {}
            for keyword in (row.get("keywords") or []):
                needle = norm(keyword)
                matches = [
                    (doc, meta) for doc, meta in corpus if needle and needle in norm(doc)
                ]
                corpus_hits[str(keyword)] = {
                    "count": len(matches),
                    "first": (
                        (matches[0][1].get("loc") or matches[0][1].get("source") or "")
                        if matches else ""
                    ),
                }
            needles = [norm(keyword) for keyword in (row.get("keywords") or []) if norm(keyword)]
            all_matches = [
                (doc, meta) for doc, meta in corpus
                if needles and all(needle in norm(doc) for needle in needles)
            ]
            all_context = ""
            all_location = ""
            if all_matches:
                doc, meta = all_matches[0]
                compact = re.sub(r"\s+", " ", str(doc or "")).strip()
                positions = [compact.casefold().find(str(keyword).casefold())
                             for keyword in (row.get("keywords") or [])]
                positions = [position for position in positions if position >= 0]
                start = max(0, (min(positions) if positions else 0) - 80)
                all_context = compact[start:start + 500]
                all_location = str(meta.get("loc") or meta.get("source") or "")
            anchor_ranks = {}
            for anchor in anchors(str(row["question"])):
                matches = [(doc, meta) for doc, meta in corpus if norm(anchor) in norm(doc)]
                rank = next((idx for idx, (_, meta) in enumerate(matches, 1)
                             if str(meta.get("loc") or meta.get("source") or "") == all_location), 0)
                anchor_ranks[anchor] = {"count": len(matches), "support_rank": rank}
            variants = {}
            for hybrid in (False, True):
                payload = backend.retrieve(
                    str(row["question"]),
                    libraries=[str(row["library_id"])],
                    hybrid=hybrid,
                    limit=8,
                )
                sources = list(payload.get("sources") or [])
                combined = norm("\n".join(
                    str(source.get("snippet") or "") for source in sources))
                variants["hybrid" if hybrid else "vector"] = {
                    "keyword_in_top8": {
                        str(keyword): norm(keyword) in combined
                        for keyword in (row.get("keywords") or [])
                    },
                    "top3": [{
                        "label": source.get("label"),
                        "distance": source.get("distance"),
                        "snippet": source.get("snippet"),
                    } for source in sources[:3]],
                }
            reports.append({
                "book": row.get("book"),
                "question": row.get("question"),
                "outcome": row.get("outcome"),
                "keywords": row.get("keywords") or [],
                "corpus_hits": corpus_hits,
                "all_match": {"count": len(all_matches), "location": all_location,
                              "context": all_context},
                "anchor_ranks": anchor_ranks,
                "retrieval": variants,
            })
        for index, report in enumerate(reports, 1):
            print(f"[{index}] {report['book']} | {report['outcome']} | {report['question']}")
            corpus = ", ".join(
                f"{key}={value['count']}@{value['first']}"
                for key, value in report["corpus_hits"].items()
            )
            print(f"  corpus: {corpus}")
            if report["all_match"]["count"]:
                print(f"  exact-doc: {report['all_match']['location']} | "
                      f"{report['all_match']['context']}")
            print("  anchors: " + ", ".join(
                f"{key}={value['count']}/support#{value['support_rank']}"
                for key, value in report["anchor_ranks"].items()
            ))
            for variant in ("vector", "hybrid"):
                result = report["retrieval"][variant]
                flags = ", ".join(
                    f"{key}={'Y' if value else 'N'}"
                    for key, value in result["keyword_in_top8"].items()
                )
                top = " | ".join(
                    f"{source['label']}@{source['distance']}"
                    for source in result["top3"]
                )
                print(f"  {variant}: {flags}; top3={top}")
        return 0
    finally:
        backend.close()


if __name__ == "__main__":
    raise SystemExit(main())
