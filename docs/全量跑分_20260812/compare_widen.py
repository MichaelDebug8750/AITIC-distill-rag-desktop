# -*- coding: utf-8 -*-
"""放宽拒答正则的配对复算。

旧版只数“编造→拒答正确”和“命中→过度拒答”两个方向，漏掉反向回补，
曾输出与聚合指标矛盾的 14:46 判决。现统一使用 eval_compare 的完整净额。
"""
import os
import sys

from eval_compare import load_rows, print_summary, summary_pair

HERE = os.path.dirname(os.path.abspath(__file__))
base = load_rows(os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "final_rows.jsonl"))
arm = load_rows(os.path.join(HERE, sys.argv[2] if len(sys.argv) > 2 else "widen_rows.jsonl"))
print_summary(summary_pair(base, arm), "窄正则", "放宽正则")

