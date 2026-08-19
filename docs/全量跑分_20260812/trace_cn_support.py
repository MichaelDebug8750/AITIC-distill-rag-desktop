# -*- coding: utf-8 -*-
"""Trace Chinese over-refusals before and after the semantic support guard.

This diagnostic runs the real in-process ``/api/ask`` implementation and records:

* the answer selected by the multi-round agent before final safety handling;
* every claim passed to the support verifier;
* the verifier's raw structured responses;
* the final audit/verdicts and delivered answer.

It does not change production behavior.  Run from any directory with the project
Python environment; output is overwritten on each run so the code/config snapshot
cannot be confused with an older trace.
"""
import io
import json
import os
import sys
import time


SP = os.path.dirname(os.path.abspath(__file__))
CODE = r"E:\Ollama_test_beta\code"
OUT = os.path.join(SP, "cn_support_trace.jsonl")
sys.path.insert(0, CODE)
os.chdir(CODE)

import main as M  # noqa: E402
import webui  # noqa: E402


QUESTIONS = [
    "舍克勒是什么？它的重量是多少？",
    "“商人”这个词的语源是什么？",
    "英语中 capital（资本）一词的语源是什么？",
    "risk（风险）一词的来源和本意是什么？",
    "银行的英文 bank 是怎么来的？",
    "谁被称作“复式记账法之父”？他写了什么书？",
    "世界最早的纸币是什么？出现在哪里？",
    "汇票和支票有什么区别？",
    "欧洲最早的中央银行是怎么产生的？",
    "成吉思汗创立的军事组织制度叫什么？",
    "蒙古帝国中被称作“斡脱”的是什么人？",
]


registry = webui._read_registry()
library_id = next(
    (item["id"] for item in registry.get("libraries", [])
     if item.get("name") == "简明世界经济史"),
    None,
)
if not library_id:
    raise SystemExit("找不到知识库：简明世界经济史")


model_calls = []
finalizer_events = []
original_model_call = webui._support_model_call
original_finalizer = webui._finalize_agent_answer


def traced_model_call(prompt):
    out = original_model_call(prompt)
    model_calls.append({
        "prompt": prompt,
        "response": out.get("response", "") if hasattr(out, "get") else "",
        "prompt_eval_count": out.get("prompt_eval_count", 0) if hasattr(out, "get") else 0,
        "eval_count": out.get("eval_count", 0) if hasattr(out, "get") else 0,
    })
    return out


def traced_finalizer(answer, packed_idx, metas, packed):
    del model_calls[:]
    before_claims = webui._claim_evidence_map(answer, packed_idx, metas, packed)
    suspicious = [
        {"id": idx, **claim}
        for idx, claim in enumerate(before_claims)
        if webui._claim_needs_support_check(claim)
    ]
    result = original_finalizer(answer, packed_idx, metas, packed)
    final_answer, cite_check, final_claims, audit, tokens = result
    finalizer_events.append({
        "pre_guard_answer": answer,
        "pre_guard_abstained": M.is_abstain(answer),
        "pre_guard_claims": before_claims,
        "suspicious_claims": suspicious,
        "packed_blocks": list(webui._support_blocks(packed_idx, metas, packed).values()),
        "verifier_calls": list(model_calls),
        "final_answer": final_answer,
        "final_abstained": M.is_abstain(final_answer),
        "final_claims": final_claims,
        "cite_check": cite_check,
        "support_audit": audit,
        "guard_tokens": tokens,
    })
    return result


webui._support_model_call = traced_model_call
webui._finalize_agent_answer = traced_finalizer


with io.open(OUT, "w", encoding="utf-8") as handle:
    for pos, question in enumerate(QUESTIONS, 1):
        del finalizer_events[:]
        started = time.time()
        error = ""
        try:
            response = webui.api_ask({
                "question": question,
                "libraries": [library_id],
                "mode": "auto",
                "style": "standard",
                "extend": False,
                "history": [],
            })
        except Exception as exc:
            response = {}
            error = repr(exc)
        event = finalizer_events[-1] if finalizer_events else {}
        row = {
            "question": question,
            "elapsed": round(time.time() - started, 2),
            "error": error,
            "rounds": ((response.get("agent") or {}).get("rounds")
                       if hasattr(response, "get") else None),
            "delivered_answer": response.get("answer", "") if hasattr(response, "get") else "",
            "trace": event,
        }
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        audit = event.get("support_audit") or {}
        print("%2d/%d refused=%-5s pre=%-5s pruned=%-2s unknown=%-2s %.1fs %s" % (
            pos, len(QUESTIONS),
            bool(event.get("final_abstained")), bool(event.get("pre_guard_abstained")),
            audit.get("pruned", 0), audit.get("unknown", 0),
            row["elapsed"], question,
        ), flush=True)

print("TRACE=" + OUT, flush=True)
