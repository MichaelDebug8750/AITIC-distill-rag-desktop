# -*- coding: utf-8 -*-
"""固定同一检索上下文，比较 WebUI 原始生成在无 seed / 固定 seed 下的重复性。

本脚本只测 ``_finalize_agent_answer`` 之前的主答案生成，不把检索、逐句核验或评分器
波动混进来。默认从已完成的双遍结果中选择默认臂发生 outcome 翻转的题；运行期间不要
同时跑其他模型实验。

用法：``seed_repeat_probe.py [rows.jsonl] [out.jsonl] [题数] [每臂重复数]``
"""
from __future__ import print_function

import hashlib
import io
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))

import webui  # noqa: E402
from eval_compare import normalize_book  # noqa: E402


DEFAULT_ROWS = os.path.join(HERE, "focus_floor_en_repaired_20260816_rows.jsonl")
DEFAULT_OUT = os.path.join(HERE, "seed_repeat_probe_20260816_rows.jsonl")
ROWS = os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_ROWS
OUT = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUT
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 8
REPEATS = int(sys.argv[4]) if len(sys.argv) > 4 else 3
SEED = 20260816


def load_jsonl(path):
    with io.open(path, encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select_cases(rows):
    grouped = {}
    for row in rows:
        if row.get("arm") != "A":
            continue
        key = (normalize_book(row.get("book")), row.get("question"))
        grouped.setdefault(key, []).append(row)
    flipped = []
    for key, group in grouped.items():
        by_pass = {int(row.get("pass")): row for row in group if row.get("pass") in (1, 2)}
        if len(by_pass) != 2 or by_pass[1].get("outcome") == by_pass[2].get("outcome"):
            continue
        row = by_pass[1]
        # 极短关键词串最易受评分边界影响；定义题与库外探针更能代表产品行为。
        priority = {"answerable": 0, "unanswerable": 1, "fuzzy_desc": 2, "fuzzy_kw": 3}.get(
            row.get("type"), 4)
        flipped.append((priority, key, row))
    flipped.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in flipped[:max(1, LIMIT)]]


def library_ids():
    registry = webui._libraries_payload().get("libraries") or []
    out = {}
    for item in registry:
        for value in (item.get("name"), item.get("source")):
            if value:
                out.setdefault(normalize_book(value), item.get("id"))
    return out


def make_prompt(row, library_id):
    question = row["question"]
    docs, metas, dists, _libraries = webui._retrieve_selected(
        question, [library_id], False, None)
    packed, packed_idx = webui._pack_agent(
        docs, metas, question, webui.M.CONTEXT_BUDGET)
    context = webui._labeled_context(packed, packed_idx, metas)
    prompt = webui._agent_prompt(
        context, question, packed_idx, metas, [], False,
        webui._response_preference("standard", ""),
        webui._evidence_looks_present(dists),
    )
    return prompt, min([x for x in dists if isinstance(x, (int, float))], default=None)


def generate(prompt, seed):
    options = {"temperature": webui.M.TEMPERATURE,
               "num_predict": webui._web_num_predict("standard")}
    if seed is not None:
        options["seed"] = int(seed)
    started = time.time()
    out = webui.M._generate(webui.M.LLM_MODEL, prompt, options=options)
    text = webui.M._strip_think(str(out.get("response") or "")).strip()
    return text, round(time.time() - started, 3)


def main():
    if not 1 <= LIMIT <= 50 or not 2 <= REPEATS <= 6:
        raise SystemExit("题数须为 1..50，每臂重复数须为 2..6")
    cases = select_cases(load_jsonl(ROWS))
    if not cases:
        raise SystemExit("输入结果中没有完整双遍的默认臂 outcome 翻转题")
    ids = library_ids()
    records = []
    for case_index, row in enumerate(cases, 1):
        library_id = ids.get(normalize_book(row.get("book")))
        if not library_id:
            raise SystemExit("找不到知识库：%s" % row.get("book"))
        prompt, best_distance = make_prompt(row, library_id)
        # 每轮交换先后，避免固定 seed 臂总是在相同热状态下运行。
        for repeat in range(1, REPEATS + 1):
            arms = (("seed", SEED), ("none", None)) if repeat % 2 else (
                ("none", None), ("seed", SEED))
            for arm, seed in arms:
                text, elapsed = generate(prompt, seed)
                records.append({
                    "case": case_index, "repeat": repeat, "arm": arm, "seed": seed,
                    "book": row.get("book"), "question": row.get("question"),
                    "type": row.get("type"), "best_distance": best_distance,
                    "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    "answer_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "answer": text, "elapsed": elapsed,
                })
                print("%2d/%d repeat=%d arm=%s chars=%d" %
                      (case_index, len(cases), repeat, arm, len(text)), flush=True)
    with io.open(OUT, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    for arm in ("none", "seed"):
        groups = {}
        for record in records:
            if record["arm"] == arm:
                groups.setdefault(record["case"], set()).add(record["answer_sha256"])
        exact = sum(len(values) == 1 for values in groups.values())
        print("arm=%s exact_repeatable=%d/%d unique_outputs=%d" %
              (arm, exact, len(groups), sum(len(values) for values in groups.values())))
    print("WROTE %s" % OUT)


if __name__ == "__main__":
    main()
