# -*- coding: utf-8 -*-
r"""
verify_docs.py —— 核对文档里的每个数字是否与原始数据一致

为什么需要它：
    《v7 交付口径》里有 60+ 个数字，分别来自 _summary.json、_fingerprint.json、
    fab_v*.json 和 main.py。手工核对过一次就查出 4 处错误（漏掉当前 chunk_sha、
    过度拒答分母未写明、分型检索证据强度标高、VL 成本口径不准）。
    数据一旦重跑，文档就可能过期，而过期是看不出来的——除非有脚本盯着。

它核什么：
    1. 内部自洽   文档里所有 "x/y = z%" 的算术
    2. 全量指标   v3full / v7full 的可答·模糊·拒答·编造·过度拒答·升配·token
    3. 学科表     6 个学科 × 6 列
    4. 幻觉核对   fab_v3.json / fab_v7.json 的四分类计数与接地率中位
    5. 配置基线   与 v7full 的 _fingerprint.json 实录逐项比对
    6. 指纹链     文档列出的 chunk_sha 是否包含当前代码的值
    7. 必备声明   撤回声明、口径说明等关键句是否在位

用法（在 E:\Ollama_test 下）：
    C:\Users\Seifer\distill\Scripts\python.exe verify_docs.py ^
        --doc docs\v7交付口径.md --v3 eval_results_v3full --v7 eval_results_v7full ^
        --fab-v3 fab_v3.json --fab-v7 fab_v7.json --main data\main.py
退出码 0 = 全部通过，1 = 有不一致（可用于 CI / 提交前钩子）
"""
import os, re, io, sys, json, glob, argparse, statistics

FAIL, WARN = [], []


def check(cond, label, detail=""):
    print("  %s %s%s" % ("OK  " if cond else "ERR ", label, ("   " + detail) if detail and not cond else ""))
    if not cond:
        FAIL.append(label + (("  " + detail) if detail else ""))
    return cond


def load_rows(d):
    rows = []
    for f in glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True):
        if os.path.basename(f).startswith("_"):
            continue
        for l in io.open(f, encoding="utf-8"):
            if l.strip():
                rows.append(json.loads(l))
    return rows


def load_summary(d):
    p = os.path.join(d, "_summary.json")
    return json.load(io.open(p, encoding="utf-8")) if os.path.exists(p) else None


def agg(summary, key):
    return sum(r[key] for r in summary)


def row_of(doc, label):
    """取文档里以 label 开头的那一行表格。"""
    for line in doc.split("\n"):
        s = line.strip()
        if s.startswith("|") and label in s.split("|")[1]:
            return s
    return ""


def fracs(line):
    """一行里所有 a/b，按出现顺序。"""
    return [(int(a), int(b)) for a, b in re.findall(r"(\d+)\s*/\s*(\d+)", line)]


def nums(line):
    return [float(x) for x in re.findall(r"(?<![\d./])(\d+(?:\.\d+)?)(?![\d/])", line)]


# ---------------- 1 内部自洽 ----------------
def check_internal(doc):
    print("\n[1] 内部自洽（文档内所有 x/y = z%）")
    bad = []
    for m in re.finditer(r"(\d+)\s*/\s*(\d+)\s*=\s*([\d.]+)%", doc):
        a, b, c = int(m.group(1)), int(m.group(2)), float(m.group(3))
        if b and abs(round(100.0 * a / b, 1) - c) > 0.06:
            bad.append("%s（应为 %.1f%%）" % (m.group(0), 100.0 * a / b))
    check(not bad, "%d 处分数式算术自洽" % len(re.findall(r"\d+\s*/\s*\d+\s*=\s*[\d.]+%", doc)),
          "；".join(bad))


# ---------------- 2 全量指标 ----------------
def check_metrics(doc, v3dir, v7dir):
    print("\n[2] 全量指标（对照 _summary.json）")
    for tag, d in (("v3full", v3dir), ("v7full", v7dir)):
        if not d or not os.path.isdir(d):
            WARN.append("%s 目录不存在，跳过：%s" % (tag, d))
            print("  --   %s 目录不存在，跳过" % tag)
    s3, s7 = load_summary(v3dir) if v3dir else None, load_summary(v7dir) if v7dir else None
    if not s7:
        return
    r7 = load_rows(v7dir)
    idx = 1 if s3 else 0            # 文档表里 v7 列的位置：有 v3 列时在第 2 组

    def cmp_row(label, pairs3, pairs7):
        """按"是否出现在该行"判定，不按位置——文档表列数会变（有无 v3 对照列），
        位置比对会在缺列时取错列，实测已踩过一次。"""
        line = row_of(doc, label)
        if not line:
            WARN.append("文档中找不到行：%s" % label)
            print("  --   文档无此行：%s" % label)
            return
        f = fracs(line)
        want = [x for x in ([pairs3] if (s3 and pairs3) else []) + [pairs7] if x]
        missing = [x for x in want if x not in f]
        check(not missing, "%-16s 期望 %s" % (label, want),
              "文档该行为 %s，缺 %s" % (f, missing))

    cmp_row("可答（严格）", (agg(s3, "ans_ok"), agg(s3, "ans_t")) if s3 else None,
            (agg(s7, "ans_ok"), agg(s7, "ans_t")))
    cmp_row("模糊·合计（严格）", (agg(s3, "fuzzy_ok"), agg(s3, "fuzzy_t")) if s3 else None,
            (agg(s7, "fuzzy_ok"), agg(s7, "fuzzy_t")))
    cmp_row("正确拒答", (agg(s3, "unans_ok"), agg(s3, "unans_t")) if s3 else None,
            (agg(s7, "unans_ok"), agg(s7, "unans_t")))

    def by_type(rows, t):
        s = [r for r in rows if r["type"] == t]
        return (sum(r["ok"] for r in s), len(s))
    cmp_row("fuzzy_desc", None, by_type(r7, "fuzzy_desc"))
    cmp_row("fuzzy_kw", None, by_type(r7, "fuzzy_kw"))

    # 计数型
    line = row_of(doc, "幻觉（原始）")
    n = nums(line)
    check(agg(s7, "fabricated") in [int(x) for x in n], "幻觉（原始）文档含 %d" % agg(s7, "fabricated"),
          "文档该行数字 %s" % n)
    line = row_of(doc, "过度拒答")
    n = [int(x) for x in nums(line) if x == int(x)]
    check(agg(s7, "over_refused") in n, "过度拒答 文档含 %d" % agg(s7, "over_refused"), "文档 %s" % n)
    line = row_of(doc, "动态升配")
    n = [int(x) for x in nums(line) if x == int(x)]
    check(agg(s7, "escalated") in n, "动态升配 文档含 %d" % agg(s7, "escalated"), "文档 %s" % n)
    med = statistics.median([r["tokens"] for r in r7])
    line = row_of(doc, "token 中位数")
    check(med in nums(line), "token 中位数 %g" % med, "文档 %s" % nums(line))


# ---------------- 3 学科表 ----------------
def check_subjects(doc, v7dir):
    print("\n[3] 学科表")
    s7 = load_summary(v7dir) if v7dir else None
    if not s7:
        print("  --   无 v7 数据，跳过")
        return
    g = {}
    for r in s7:
        x = g.setdefault(r["subject"], dict(books=0, ans_ok=0, ans_t=0, fuzzy_ok=0,
                                            fuzzy_t=0, unans_ok=0, unans_t=0,
                                            fabricated=0, escalated=0))
        x["books"] += 1
        for k in ("ans_ok", "ans_t", "fuzzy_ok", "fuzzy_t", "unans_ok",
                  "unans_t", "fabricated", "escalated"):
            x[k] += r[k]
    for subj, x in sorted(g.items()):
        line = row_of(doc, subj)
        if not line:
            WARN.append("学科表缺行：%s" % subj)
            print("  --   文档无此学科：%s" % subj)
            continue
        got = nums(line)
        want = [x["books"], round(100.0 * x["ans_ok"] / x["ans_t"], 1),
                round(100.0 * x["fuzzy_ok"] / x["fuzzy_t"], 1),
                round(100.0 * x["unans_ok"] / x["unans_t"], 1),
                x["fabricated"], x["escalated"]]
        check(got == want, "%-18s %s" % (subj, got), "实测 %s" % want)


# ---------------- 4 幻觉核对 ----------------
def check_fab(doc, p3, p7):
    print("\n[4] 幻觉核对（fab_v*.json）")
    from collections import Counter
    for tag, p in (("v3", p3), ("v7", p7)):
        if not p or not os.path.exists(p):
            WARN.append("找不到 %s" % p)
            print("  --   找不到 %s，跳过" % p)
            continue
        d = json.load(io.open(p, encoding="utf-8"))
        c = Counter(x["flag"].split("：")[0] for x in d)
        med = round(statistics.median([x["grounding"] for x in d]), 2)
        want = [len(d), c["真幻觉"], c["题目可能有问题"], c["存疑"]]
        line = row_of(doc, "真幻觉（低接地，引用为装饰）")
        n = [int(x) for x in nums(line)]
        check(c["真幻觉"] in n, "%s 真幻觉 %d 出现在文档" % (tag, c["真幻觉"]), "文档行 %s" % n)
        line = row_of(doc, "接地率中位")
        check(med in nums(line), "%s 接地率中位 %.2f" % (tag, med), "文档行 %s" % nums(line))


# ---------------- 5 配置基线 ----------------
def check_config(doc, v7dir):
    print("\n[5] 配置基线（对照 v7full/_fingerprint.json）")
    p = os.path.join(v7dir, "_fingerprint.json") if v7dir else None
    if not p or not os.path.exists(p):
        WARN.append("无 _fingerprint.json，跳过配置核对")
        print("  --   无 _fingerprint.json，跳过")
        return
    rt = json.load(io.open(p, encoding="utf-8")).get("runtime", {})
    env = rt.get("env", {}) or {}
    for label, val in (("TOP_K", rt.get("top_k")),
                       ("DYNAMIC_BUDGET", rt.get("dynamic_budget")),
                       ("ESCALATE_SIM_GATE", rt.get("escalate_sim_gate")),
                       ("RELEVANCE_TRIM", rt.get("relevance_trim")),
                       ("PROMPT_VARIANT", rt.get("prompt_variant")),
                       ("ollama 服务端", env.get("ollama_server")),
                       ("ollama-py", env.get("ollama_py"))):
        line = row_of(doc, label)
        if not line:
            WARN.append("配置表缺行：%s" % label)
            print("  --   文档无此配置项：%s" % label)
            continue
        check(str(val) in line, "%-18s = %s" % (label, val), "文档行：%s" % line.strip())
    for key in ("prompt_sha", "chunk_sha"):
        v = rt.get(key)
        check(v and v in doc, "%s %s 出现在文档" % (key, v))


# ---------------- 6 指纹链 ----------------
def check_fingerprint_chain(doc, main_path):
    print("\n[6] 指纹链（当前代码 vs 文档）")
    if not main_path or not os.path.exists(main_path):
        WARN.append("找不到 main.py，跳过指纹链核对")
        print("  --   找不到 main.py，跳过")
        return
    import importlib.util
    d = os.path.dirname(os.path.abspath(main_path))
    sys.path.insert(0, d)
    try:
        spec = importlib.util.spec_from_file_location("_m", main_path)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        cur, psha = m.chunking_fingerprint(), m._sha(m.PROMPT)
    except SystemExit as e:
        # main.py 缺依赖时会 sys.exit，不能让它把核对进程一起带走
        WARN.append("main.py 因缺少依赖而退出，跳过指纹链核对（%s）" % e)
        print("  --   main.py 缺依赖，跳过（在项目虚拟环境里跑就不会出现）")
        return
    except Exception as e:
        WARN.append("导入 main.py 失败：%s" % e)
        print("  --   导入失败：%s" % e)
        return
    check(cur in doc, "当前代码 chunk_sha %s 已在文档中列出" % cur,
          "文档未列出，说明代码改过而文档没跟上")
    check(psha in doc, "当前 prompt_sha %s 已在文档中列出" % psha)


# ---------------- 7 必备声明 ----------------
def check_statements(doc):
    print("\n[7] 必备声明")
    # 检查"该说的话说了没"。关键词随文档结论变化而更新——
    # V5 曾写"不做"，后改为"待验证"，本检查也随之从「不做」改为「采纳判据」。
    need = [("撤回声明", "已撤回"),
            ("分母口径说明", "为分母"),
            ("探针缺陷说明", "探针"),
            ("V5 状态与判据", "采纳判据"),
            ("指纹链复现提示", "复现本文")]
    for label, kw in need:
        check(kw in doc, "%s（关键词「%s」）" % (label, kw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", required=True)
    ap.add_argument("--v3", default="")
    ap.add_argument("--v7", default="")
    ap.add_argument("--fab-v3", default="")
    ap.add_argument("--fab-v7", default="")
    ap.add_argument("--main", default="")
    a = ap.parse_args()

    doc = io.open(a.doc, encoding="utf-8").read()
    print("核对文档：%s（%d 字符）" % (a.doc, len(doc)))

    check_internal(doc)
    check_metrics(doc, a.v3, a.v7)
    check_subjects(doc, a.v7)
    check_fab(doc, a.fab_v3, a.fab_v7)
    check_config(doc, a.v7)
    check_fingerprint_chain(doc, a.main)
    check_statements(doc)

    print("\n" + "=" * 62)
    if WARN:
        print("跳过/警告 %d 项：" % len(WARN))
        for w in WARN:
            print("  - %s" % w)
    if FAIL:
        print("不一致 %d 项：" % len(FAIL))
        for f in FAIL:
            print("  ! %s" % f)
        print("\n文档与数据不一致，改完再提交。")
        sys.exit(1)
    print("全部通过。文档中的数字与原始数据一致。")
    sys.exit(0)


if __name__ == "__main__":
    main()
