# -*- coding: utf-8 -*-
"""对照 question 单键旧算法与复合键结果；只报告差异，不用固定噪声自动采纳。"""
from __future__ import print_function

import json
import os

from eval_compare import load_rows, summary_pair

HERE = os.path.dirname(os.path.abspath(__file__))
ARMS = [
    ("放宽正则", "final_rows.jsonl", "widen_rows.jsonl"),
    ("拒答采纳", "final_rows.jsonl", "adopt_rows.jsonl"),
    ("关闭下限", "final_rows.jsonl", "floor_rows.jsonl"),
    ("撤文风闸门", "final_rows.jsonl", "style_rows.jsonl"),
    ("校验轮新措辞", "final_rows.jsonl", "verify_rows.jsonl"),
    ("混合检索", "reg_rows.jsonl", "hyb_rows.jsonl"),
]


def load_question_overwrite(name):
    out = {}
    with open(os.path.join(HERE, name), encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                out[row["question"]] = row
    return out


def net(base, arm):
    common = set(base) & set(arm)
    hit = sum(arm[key].get("outcome") == "命中" for key in common) - \
          sum(base[key].get("outcome") == "命中" for key in common)
    fab = sum(arm[key].get("outcome") == "编造" for key in common) - \
          sum(base[key].get("outcome") == "编造" for key in common)
    return len(common), hit - 2 * fab


print("%-14s %-20s %-22s %s" % ("臂", "旧 question 单键", "复合键", "净值是否一致"))
for label, base_name, arm_name in ARMS:
    old_n, old_net = net(load_question_overwrite(base_name), load_question_overwrite(arm_name))
    summary = summary_pair(load_rows(os.path.join(HERE, base_name)),
                           load_rows(os.path.join(HERE, arm_name)))
    print("%-14s n=%-4d 净值 %+-4d    n=%-4d 净值 %+-4d      %s" %
          (label, old_n, old_net, len(summary["common"]), summary["net"],
           "一致" if old_net == summary["net"] else "不同"))
print("\n旧键仅用于复现历史缺陷；正式结论必须使用复合键。这里不使用固定噪声阈值自动采纳。")
