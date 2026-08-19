# -*- coding: utf-8 -*-
"""逐条展示条件混合相对纯向量新增/修复的编造，避免净数掩盖失败形态。"""
from __future__ import print_function

import os

from eval_compare import key_label, load_rows

HERE = os.path.dirname(os.path.abspath(__file__))


def preview(row):
    return " ".join(str(row.get("answer") or "").split())[:360]


def show(base_name, arm_name, label):
    base = load_rows(os.path.join(HERE, base_name))
    arm = load_rows(os.path.join(HERE, arm_name))
    if len(base) != 1007 or len(arm) != 1007 or set(base) != set(arm):
        raise ValueError("%s 两臂必须各含 1007 个相同复合键" % label)
    new = [key for key in base if base[key].get("outcome") == "拒答正确"
           and arm[key].get("outcome") == "编造"]
    cured = [key for key in base if base[key].get("outcome") == "编造"
             and arm[key].get("outcome") == "拒答正确"]
    print("\n=== %s：新增 %d / 修复 %d / 净增 %+d ===" %
          (label, len(new), len(cured), len(new) - len(cured)))
    for title, keys in (("新增编造", new), ("修复编造", cured)):
        print("\n-- %s --" % title)
        for key in sorted(keys):
            print("%s\n  base: %s\n  arm : %s" %
                  (key_label(key), preview(base[key]), preview(arm[key])))


def main():
    show("cleanbase_rows.jsonl", "hybgate_rows.jsonl", "原始回包")
    show("cleanbase_finalcode_replay.jsonl", "hybgate_finalcode_replay.jsonl",
         "当前确定性拒答规则回放")
    print("\n失败形态需人工审阅；本脚本不凭关键词自动宣称‘碰撞型’或替用户改变默认。")


if __name__ == "__main__":
    main()
