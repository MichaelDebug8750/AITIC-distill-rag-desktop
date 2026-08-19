# -*- coding: utf-8 -*-
"""比较修复后同构建的纯向量/条件混合臂，不自动替用户做产品取舍。"""
from __future__ import print_function

import collections
import os

from eval_compare import load_rows, print_summary, summary_pair

HERE = os.path.dirname(os.path.abspath(__file__))
EXPECTED_ROWS = 1007


def load_complete(name):
    path = os.path.join(HERE, name)
    rows = load_rows(path)
    if len(rows) != EXPECTED_ROWS:
        raise ValueError("%s 应为 %d 个唯一复合键，实际 %d" %
                         (name, EXPECTED_ROWS, len(rows)))
    failed = [key for key, row in rows.items()
              if not isinstance(row.get("status"), int)
              or not 200 <= row.get("status") < 300
              or row.get("outcome") == "请求失败" or row.get("error")]
    if failed:
        raise ValueError("%s 含 %d 条请求失败/非 2xx，拒绝进入指标" % (name, len(failed)))
    return rows


def counts(rows, keys):
    return collections.Counter(rows[key].get("outcome") for key in keys)


def print_absolute(label, rows, keys):
    tally = counts(rows, keys)
    answerable = sum(1 for key in keys if rows[key].get("expect") != "abstain")
    unanswerable = len(keys) - answerable
    print("%-18s 命中 %3d/%3d = %5.1f%%  编造 %2d/%3d = %4.1f%%  过拒 %d" %
          (label, tally["命中"], answerable,
           100.0 * tally["命中"] / answerable if answerable else 0.0,
           tally["编造"], unanswerable,
           100.0 * tally["编造"] / unanswerable if unanswerable else 0.0,
           tally["过度拒答"]))


def compare(base_name, arm_name, label):
    base, arm = load_complete(base_name), load_complete(arm_name)
    summary = summary_pair(base, arm)
    if len(summary["common"]) != EXPECTED_ROWS or summary["base_only"] or summary["arm_only"]:
        raise ValueError("%s 两臂没有完整一一配对" % label)
    print("\n=== %s ===" % label)
    print_absolute("纯向量", base, summary["common"])
    print_absolute("条件混合", arm, summary["common"])
    print_summary(summary, base_name, arm_name)
    return summary


def main():
    raw = compare("cleanbase_rows.jsonl", "hybgate_rows.jsonl", "原始回包")
    replay = compare("cleanbase_finalcode_replay.jsonl",
                     "hybgate_finalcode_replay.jsonl", "当前确定性拒答规则回放")
    print("\n原始净额：命中 %+d / 编造 %+d / 净值 %+d" %
          (raw["deltas"]["命中"], raw["deltas"]["编造"], raw["net"]))
    print("回放净额：命中 %+d / 编造 %+d / 净值 %+d" %
          (replay["deltas"]["命中"], replay["deltas"]["编造"], replay["net"]))
    print("注意：两臂同构建但非同时段空白对照；这里报告完整配对事实，不把固定噪声常数"
          "或净值阈值包装成自动采纳结论。")


if __name__ == "__main__":
    main()
