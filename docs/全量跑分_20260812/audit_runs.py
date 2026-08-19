# -*- coding: utf-8 -*-
"""盘点所有跑分产物的完成度，以及它们跑在哪个构建上。

关键：cn2h / cn2kw 跑在 _usable_dists 修复之前（那时 None 距离会抛异常、
流程绕过拒答判断），所以那两臂的数据**不能代表当前构建**。
"""
import glob
import io
import json
import os
import time

SP = os.path.dirname(os.path.abspath(__file__))
os.chdir(SP)

FIX_TS = None
for p in ("E:\\Ollama_test_beta\\code\\webui.py",):
    if os.path.exists(p):
        FIX_TS = os.path.getmtime(p)

print("%-20s %6s %7s %-14s %s" % ("文件", "行数", "唯一题", "最后写入", "状态"))
for p in sorted(glob.glob("*rows*.jsonl")):
    qs, n = set(), 0
    for line in io.open(p, encoding="utf-8"):
        if line.strip():
            n += 1
            try:
                qs.add(json.loads(line)["question"])
            except Exception:
                pass
    mt = os.path.getmtime(p)
    when = time.strftime("%m-%d %H:%M", time.localtime(mt))
    if p.startswith("cn_"):
        target = 24
    elif p.startswith("cn2"):
        target = 30
    else:
        target = 972
    st = "完成" if len(qs) >= target else "**未完成**"
    print("%-20s %6d %7d %-14s %s" % (p, n, len(qs), when, st))

print("\n中文各臂跑在什么构建上（webui.py 最后修改 %s）："
      % time.strftime("%m-%d %H:%M", time.localtime(FIX_TS)))
for p in sorted(glob.glob("cn*rows*.jsonl")):
    mt = os.path.getmtime(p)
    print("  %-20s %s  %s" % (p, time.strftime("%m-%d %H:%M", time.localtime(mt)),
                              "跑在当前构建之前 → 结论不可用" if mt < FIX_TS else "当前构建"))
