# -*- coding: utf-8 -*-
"""全量评测的稳定身份与配对净额工具。

同一个库外问题会被复用于多本教材。旧脚本只用 ``question`` 作字典键，
1007 行会静默缩成 972 个键，35 个“同题、不同书”的样本被覆盖。
本模块统一用 ``(规范化书名, 题目)``，并在复合键仍重复时立即报错。

配对结论必须统计全部迁移方向。``summary_pair`` 同时从聚合计数和迁移矩阵
计算净额，两者对不上就抛错，避免再次出现“判定行与聚合指标互相矛盾”。
"""
from __future__ import print_function

import collections
import io
import json
import os
import re
import unicodedata


def normalize_book(value):
    """去扩展名/标点并统一 Unicode；不做编辑距离等模糊猜测。"""
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    text = os.path.splitext(os.path.basename(text))[0]
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text).casefold()


def row_key(row):
    """返回稳定复合键；缺书名或题目时拒绝静默降级为不安全键。"""
    question = unicodedata.normalize("NFKC", str((row or {}).get("question") or "")).strip()
    book = normalize_book((row or {}).get("book") or (row or {}).get("library"))
    if not question or not book:
        raise ValueError("评测行缺少 book/question，不能安全配对: %r" % (row,))
    return book, question


def load_rows(path):
    """读取 JSONL，并拒绝复合键重复；不再用后写覆盖前写。"""
    rows = {}
    first_line = {}
    for line_no, line in enumerate(io.open(path, encoding="utf-8"), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = row_key(row)
        if key in rows:
            raise ValueError(
                "%s:%d 与第 %d 行复合键重复: %s / %s" %
                (path, line_no, first_line[key], key[0], key[1]))
        rows[key] = row
        first_line[key] = line_no
    return rows


def build_question_index(rows):
    """为题集元数据建索引；同题可属于多本书，故值始终是列表。"""
    index = collections.defaultdict(list)
    for row in rows:
        question = unicodedata.normalize("NFKC", str(row.get("question") or "")).strip()
        if question:
            index[question].append(row)
    return dict(index)


def match_question_row(row, question_index):
    """把截断显示书名的跑分行确定性地映射回完整题集行。

    ``fullrun3.py`` 历史上把书名截为 28 字符，因此先做完整规范化匹配，再允许
    唯一的前缀匹配。仍有多个候选时直接报错，不猜一本书。
    """
    question = unicodedata.normalize("NFKC", str(row.get("question") or "")).strip()
    candidates = list((question_index or {}).get(question) or [])
    if not candidates:
        raise KeyError("题集里找不到问题: %s" % question)
    if len(candidates) == 1:
        return candidates[0]
    wanted = normalize_book(row.get("book") or row.get("library"))
    exact = [item for item in candidates if normalize_book(item.get("book")) == wanted]
    if len(exact) == 1:
        return exact[0]
    # 历史跑分行是把完整题集书名截短，因此只允许“完整候选以截断值开头”。
    # 反向也放行会把 ``Business Law`` 误纳入
    # ``Business Law and the Legal E...`` 的候选，制造假歧义甚至错配。
    prefix = [item for item in candidates
              if wanted and normalize_book(item.get("book")).startswith(wanted)]
    if len(prefix) == 1:
        return prefix[0]
    raise ValueError("同题跨书且书名无法唯一匹配: %s / %s" % (row.get("book"), question))


def _delta_from_matrix(migrations, target):
    incoming = sum(n for (before, after), n in migrations.items() if after == target)
    outgoing = sum(n for (before, after), n in migrations.items() if before == target)
    return incoming - outgoing


def summary_pair(base, arm, fabrication_weight=2):
    """返回完整四向配对净额，并强制做迁移矩阵交叉校验。"""
    common = sorted(set(base) & set(arm))
    migrations = collections.Counter(
        (base[key].get("outcome"), arm[key].get("outcome"))
        for key in common if base[key].get("outcome") != arm[key].get("outcome"))

    def count(rows, outcome):
        return sum(1 for key in common if rows[key].get("outcome") == outcome)

    deltas = {
        "命中": count(arm, "命中") - count(base, "命中"),
        "编造": count(arm, "编造") - count(base, "编造"),
        "过度拒答": count(arm, "过度拒答") - count(base, "过度拒答"),
        "拒答正确": count(arm, "拒答正确") - count(base, "拒答正确"),
    }
    for outcome, delta in deltas.items():
        matrix_delta = _delta_from_matrix(migrations, outcome)
        if delta != matrix_delta:
            raise AssertionError("%s 净额 %d 与迁移矩阵 %d 不一致" %
                                 (outcome, delta, matrix_delta))
    return {
        "common": common,
        "base_only": sorted(set(base) - set(arm)),
        "arm_only": sorted(set(arm) - set(base)),
        "migrations": migrations,
        "moved": sum(migrations.values()),
        "deltas": deltas,
        "fabrication_weight": fabrication_weight,
        "net": deltas["命中"] - fabrication_weight * deltas["编造"],
    }


def key_label(key):
    return "%s :: %s" % (key[0], key[1])


def print_summary(summary, base_label="基线", arm_label="实验臂"):
    """打印不带自动采纳结论的完整配对结果。"""
    common = summary["common"]
    print("配对题数 %d（仅基线 %d / 仅实验臂 %d）\n" %
          (len(common), len(summary["base_only"]), len(summary["arm_only"])))
    print("=== 全部迁移矩阵 ===")
    for (before, after), count in summary["migrations"].most_common():
        print("  %-10s -> %-10s %4d" % (before, after, count))
    rate = 100.0 * summary["moved"] / len(common) if common else 0.0
    print("  合计变动 %d / %d = %.1f%%" % (summary["moved"], len(common), rate))
    print("\n=== 净额（已与迁移矩阵交叉校验）===")
    for outcome in ("命中", "编造", "过度拒答", "拒答正确"):
        print("  %-10s %+d" % (outcome, summary["deltas"][outcome]))
    print("  净值 = 命中增量 - %d x 编造增量 = %+d" %
          (summary["fabrication_weight"], summary["net"]))
    print("\n%s vs %s：不在这里自动判定采纳。必须再与同条件、同时段的空白对照比较。" %
          (base_label, arm_label))
