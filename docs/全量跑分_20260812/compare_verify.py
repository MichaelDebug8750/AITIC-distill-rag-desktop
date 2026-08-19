# -*- coding: utf-8 -*-
"""校验轮原/新措辞的完整配对净额；不再用单向迁移自动下判决。"""
import os
import sys

from eval_compare import load_rows, print_summary, summary_pair

HERE = os.path.dirname(os.path.abspath(__file__))
base = load_rows(os.path.join(HERE, sys.argv[1] if len(sys.argv) > 1 else "final_rows.jsonl"))
arm = load_rows(os.path.join(HERE, sys.argv[2] if len(sys.argv) > 2 else "verify_rows.jsonl"))
print_summary(summary_pair(base, arm), "校验轮原措辞", "校验轮新措辞")

