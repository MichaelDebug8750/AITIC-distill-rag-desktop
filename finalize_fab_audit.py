# -*- coding: utf-8 -*-
"""把 verify_fab.py 的 v8 自动底稿固化为逐条语义审计记录。"""
import argparse
import io
import json
from collections import Counter


PROBE_DEFECT = {
    ("Current Essentials of Medicine.pdf", "What is a distorted body image?"):
        "引用页明确写有身体形状/体重知觉障碍及躯体变形障碍；答案是原文概念的合理同义转述，探针的字面零出现规则误判。",
    ("Entrepreneurship.pdf", "What is a cause-and-effect relationship?"):
        "原文术语因换行/连字符写成 cause-and effect-relationships，并给出同一定义；属于规范化漏检。",
    ("Principles of Economics.pdf", "What are the main ingredients of Neapolitan pizza dough?"):
        "引用页逐项写明 dough 使用 flour、yeast、water，答案完全由原文支持。",
    ("Principles of Economics.pdf", "What is an Americans with Disabilities Act?"):
        "引用页明确列出 Americans with Disabilities Act of 1990 及其禁止歧视、要求合理便利的定义；换行造成字面检索漏检。",
}

UNCERTAIN = {
    ("Psychology The Science of Behaviour.pdf", "What is an internal environment?"):
        "引用材料确实讨论脑调节身体生理过程，但没有把 internal environment 作为独立术语定义；答案可能是合理综合，也可能是语义拉伸。",
}


def _clean_fenced_text(text):
    """Remove source-formatting whitespace that would dirty Markdown diffs."""
    return "\n".join(line.rstrip() for line in (text or "").splitlines()).rstrip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="eval/fab_v8_auto.json")
    ap.add_argument("--out", default="eval/fab_v8_manual_audit")
    args = ap.parse_args()
    rows = json.load(io.open(args.input, encoding="utf-8"))
    if len(rows) != 50:
        raise SystemExit("预期 50 条，实际 %d 条" % len(rows))

    audited = []
    for row in rows:
        key = (row["book"], row["question"])
        item = dict(row)
        if key in PROBE_DEFECT:
            item["manual_verdict"] = "PROBE_DEFECT_SUPPORTED"
            item["manual_note"] = PROBE_DEFECT[key]
        elif key in UNCERTAIN:
            item["manual_verdict"] = "UNCERTAIN"
            item["manual_note"] = UNCERTAIN[key]
        else:
            item["manual_verdict"] = "CONFIRMED_HALLUCINATION"
            item["manual_note"] = (
                "引用来源没有定义问题中的特定术语；回答把邻近概念、普通词义或模型常识"
                "重新命名为该术语，引用不足以支撑所给定义。"
            )
        audited.append(item)

    counts = Counter(x["manual_verdict"] for x in audited)
    json.dump(audited, io.open(args.out + ".json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    with io.open(args.out + ".md", "w", encoding="utf-8") as f:
        f.write("# v8final 自动编造 50 条逐条语义审计（Codex 辅助）\n\n")
        f.write("## 结论\n\n")
        f.write("- 确认真幻觉：**%d/50**。\n" % counts["CONFIRMED_HALLUCINATION"])
        f.write("- 题集探针缺陷、答案有原文支持：**%d/50**。\n" % counts["PROBE_DEFECT_SUPPORTED"])
        f.write("- 仍存疑：**%d/50**。\n" % counts["UNCERTAIN"])
        f.write("- 以全部 1374 道不可答题为分母：确认幻觉率 **%.2f%%**；保守地把存疑也计入则为 **%.2f%%**。\n\n" %
                (100.0 * counts["CONFIRMED_HALLUCINATION"] / 1374,
                 100.0 * (counts["CONFIRMED_HALLUCINATION"] + counts["UNCERTAIN"]) / 1374))
        f.write("判据：引用页/章是否真实支持答案对问题中特定术语的定义。自动接地率只用于排序，逐条语义结论优先。\n\n")
        f.write("> 边界：本记录由 Codex 逐条复核并固化证据，不等同于项目成员人工签字。若验收明确要求人工终审，仍需项目成员逐条确认，尤其是 1 条 UNCERTAIN。\n\n")
        for i, x in enumerate(sorted(audited, key=lambda y: (y["manual_verdict"], y["book"], y["question"])), 1):
            f.write("---\n\n## %02d. %s\n\n" % (i, x["question"]))
            f.write("- 书：`%s`（%s）\n" % (x["book"], x.get("subject")))
            f.write("- 术语：`%s`\n" % (x.get("term") or "（题集为空）"))
            f.write("- 自动接地率：%.3f；字面出现：%s；逐词共现：%s\n" %
                    (x.get("grounding", 0), x.get("exact"), x.get("cooccur")))
            f.write("- **人工结论：%s**\n" % x["manual_verdict"])
            f.write("- 理由：%s\n\n" % x["manual_note"])
            answer = (x.get("answer") or "").replace("\n", " ").rstrip()
            f.write("**模型答案**\n\n> %s\n\n" % answer)
            if x.get("cited_text"):
                f.write("**引用原文摘录**\n\n```\n%s\n```\n" %
                        _clean_fenced_text(x["cited_text"]))
    print(dict(counts))


if __name__ == "__main__":
    main()
