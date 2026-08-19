# -*- coding: utf-8 -*-
"""通用净额对照；统一复用 eval_compare，且不内置跨时段噪声或自动判定。

用法：``net_compare.py [base.jsonl] [arm.jsonl]``。
同一道库外问题会用于多本教材，因此身份必须是规范化后的 ``(book, question)``；
所有迁移方向和净额交叉校验都由唯一权威实现 ``summary_pair`` 负责。
"""
from __future__ import print_function

import os
import sys

from eval_compare import load_rows, print_summary, summary_pair

HERE = os.path.dirname(os.path.abspath(__file__))
base_name = sys.argv[1] if len(sys.argv) > 1 else "final_rows.jsonl"
arm_name = sys.argv[2] if len(sys.argv) > 2 else "verify_rows.jsonl"


def resolve(name):
    """显式相对路径按调用者 cwd；裸文件名再回退到本脚本归档目录。"""
    if os.path.isabs(name) or os.path.exists(name):
        return os.path.abspath(name)
    return os.path.join(HERE, name)


base = load_rows(resolve(base_name))
arm = load_rows(resolve(arm_name))
print_summary(summary_pair(base, arm), os.path.basename(base_name), os.path.basename(arm_name))
print("\n本脚本只报告完整配对事实。噪声必须由同条件实验现场测量，"
      "不得写成跨时段常数，也不得在这里自动采纳产品改动。")
