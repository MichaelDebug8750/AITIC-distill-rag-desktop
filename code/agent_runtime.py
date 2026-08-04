# -*- coding: utf-8 -*-
"""
agent_runtime.py — 智能体对话运行时（检索 + 计算器 两工具 · 规则路由）

由 `main.py agent` 生成的智能体包（system_prompt.txt / Modelfile / run.bat）调用。
职责：加载定制 system prompt，进入对话循环，每轮按规则路由到：
  · 计算器工具：问题含可解析算式 → 白名单 AST 安全求值（不走 LLM，杜绝算错）
  · 检索工具  ：其余 → 复用 main 的 RAG（检索预算 + Citation Grounding）

计算器安全性：白名单 AST 求值，绝不使用 eval()；只允许四则/幂/取模、括号、
一元正负，以及有限数学函数（sqrt/sin/cos/log/...）与常量（pi/e）。
"""
import ast
import math
import operator
import re
import os
import sys
import argparse

# ----------------------------- 计算器：白名单 AST 求值 -----------------------------
_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS = {k: getattr(math, k) for k in
          ("sqrt", "sin", "cos", "tan", "log", "log2", "log10", "exp",
           "floor", "ceil", "factorial")}
_FUNCS.update({"abs": abs, "round": round})
_CONSTS = {"pi": math.pi, "e": math.e}


def _ev(node):
    if isinstance(node, ast.Expression):
        return _ev(node.body)
    if isinstance(node, ast.Constant):            # 数字常量
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("非法常量")
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise ValueError("非法二元运算符")
        return op(_ev(node.left), _ev(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise ValueError("非法一元运算符")
        return op(_ev(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _CONSTS:
            return _CONSTS[node.id]
        raise ValueError("未知标识符：%s" % node.id)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ValueError("非白名单函数")
        return _FUNCS[node.func.id](*[_ev(a) for a in node.args])
    raise ValueError("不支持的表达式元素：%s" % type(node).__name__)


def safe_eval(expr):
    """只对白名单 AST 求值；任何越界结构直接抛错。绝不 eval()。"""
    return _ev(ast.parse(expr, mode="eval"))


# ----------------------------- 规则路由：抽取算式 -----------------------------
# 必要条件：存在"数 运算符 数/("，避免把 "COVID-19"、"1918年" 等误判为算式
_BINARY_MATH = re.compile(r"\d\s*(\*\*|[+\-*/%])\s*[\d(]")
_EXPR_CHARS = re.compile(r"[0-9.+\-*/%() ]+")
_FUNC_CALL = re.compile(
    r"(sqrt|sin|cos|tan|log10|log2|log|exp|floor|ceil|factorial|abs|round)\s*\([^)]*\)")


def extract_expr(q):
    """从问题里抽出可安全求值的算式；抽不到返回 None（→ 走检索）。"""
    s = q.strip().replace("×", "*").replace("÷", "/").replace("^", "**")
    # 1) 纯算式片段：取最长且能安全求值、且含二元运算的候选
    for cand in sorted(_EXPR_CHARS.findall(s), key=len, reverse=True):
        cand = cand.strip()
        if len(cand) < 3 or not _BINARY_MATH.search(cand):
            continue
        try:
            safe_eval(cand)
            return cand
        except Exception:
            continue
    # 2) 数学函数调用：如 sqrt(144)、log(100)
    m = _FUNC_CALL.search(s)
    if m:
        try:
            safe_eval(m.group(0))
            return m.group(0)
        except Exception:
            pass
    return None


def route(q):
    """返回 ('calc', 表达式) 或 ('retrieve', None)。"""
    expr = extract_expr(q)
    return ("calc", expr) if expr is not None else ("retrieve", None)


# ----------------------------- 对话运行时 -----------------------------
def _answer_with_system(main, col, question, system_text, model_name=None):
    """按主程序同一检索、打包、闸门和引用口径运行生成智能体。"""
    qv = main.embed([question])[0]
    docs, metas, dists = main._retrieve(col, qv, question)

    def run_once(budget):
        if main.RELEVANCE_TRIM:
            packed, packed_idx = main._pack_relevance(docs, question, budget)
        else:
            packed, packed_idx = main._pack_truncate(docs, budget)
        context = main._labeled_context(packed, packed_idx, metas)
        out = main._generate(
            model_name or main.LLM_MODEL,
            main._format_prompt(context, question, packed_idx, metas),
            system=system_text or None,
            options={"temperature": main.TEMPERATURE, "num_predict": main.NUM_PREDICT},
        )
        toks = out.get("prompt_eval_count", 0) + out.get("eval_count", 0)
        return out["response"].strip(), toks, packed_idx

    ans, toks, packed_idx = run_once(main.CONTEXT_BUDGET)
    escalated = False
    if main.should_escalate(ans, docs, dists, main.DYNAMIC_BUDGET):
        ans2, toks2, idx2 = run_once(main.BUDGET_ESCALATED)
        toks += toks2
        escalated = True
        if not main.is_abstain(ans2):
            ans, packed_idx = ans2, idx2

    srcs = sorted({"[%s]" % main._cite_tag(metas[i])
                   for i in packed_idx if i < len(metas)})
    best_dist = min(dists) if dists else None
    tag = "  (动态升配 %d)" % main.BUDGET_ESCALATED if escalated else ""
    if best_dist is not None:
        tag += "  dist=%.4f" % best_dist
    cc = main.verify_citations(ans, packed_idx, metas)
    print("\n" + ans)
    print("\n[来源] %s  | tokens: %d%s" % ("、".join(srcs), toks, tag))
    print("[引用校验] %s" % main._cite_check_line(cc))
    return ans


def run(db_path, collection=None, system_path=None, model_name=None):
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)          # 复用同目录 main.py
    import main
    import chromadb

    system_text = ""
    if system_path and os.path.exists(system_path):
        system_text = open(system_path, encoding="utf-8").read().strip()

    if not os.path.exists(db_path):
        sys.exit("[错误] 找不到向量库：%s（先用 main.py build 建库）" % db_path)
    client = chromadb.PersistentClient(path=db_path)
    col = client.get_collection(collection or main.COLLECTION)

    print("== 智能体已启动（工具：检索 + 计算器 | 规则路由）==")
    print("   模型：%s" % (model_name or main.LLM_MODEL))
    print("   直接提问走检索问答；输入算式（如 3*(4+5)）走计算器；exit 退出。")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in ("exit", "quit", ""):
            break
        tool, expr = route(q)
        if tool == "calc":
            try:
                val = safe_eval(expr)
                print("[计算器] %s = %s" % (expr, val))
            except Exception as e:
                print("[计算器] 无法求值（%s），转检索。" % e)
                _answer_with_system(main, col, q, system_text, model_name)
        else:
            _answer_with_system(main, col, q, system_text, model_name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="智能体对话运行时（检索+计算器·规则路由）")
    ap.add_argument("--db", required=True, help="向量库路径（如 ../vectordb）")
    ap.add_argument("--collection", default=None)
    ap.add_argument("--prompt", default=None, help="system_prompt.txt 路径")
    ap.add_argument("--model", default=None, help="生成包创建的专属 Ollama 模型名")
    args = ap.parse_args()
    run(args.db, args.collection, args.prompt, args.model)
