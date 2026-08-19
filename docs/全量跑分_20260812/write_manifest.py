# -*- coding: utf-8 -*-
"""把当前进程实际加载到的开关状态落盘成 manifest。

为什么要有这个文件：`cn2kw` 那一臂因为配置只存在于启动时的 shell 环境变量里，
事后无法重建，整臂作废（§三十五）。从此每臂起跑前必须落盘。

用法：write_manifest.py <臂名>
"""
import json
import os
import sys

sys.path.insert(0, r"E:\Ollama_test_beta\code")
import webui                                        # noqa: E402

SP = os.path.dirname(os.path.abspath(__file__))
arm = sys.argv[1] if len(sys.argv) > 1 else "unknown"

cfg = {
    "arm": arm,
    "hybrid_enabled": bool(webui._hybrid_enabled(None)),
    "evidence_floor": webui._EVIDENCE_FLOOR,
    "style_gate": webui._STYLE_GATE_MAX,
    "verify_keep": bool(webui._VERIFY_KEEP),
    "adopt_abstain": bool(webui._ADOPT_ABSTAIN),
    "widen_refusal": bool(webui._WIDEN_REFUSAL),
    "kw_df_ratio_env": os.environ.get("AITIC_KW_DF_RATIO"),
    "env": {k: os.environ.get(k) for k in
            ("DISTILL_HYBRID", "AITIC_EVIDENCE_FLOOR", "AITIC_STYLE_GATE",
             "AITIC_VERIFY_KEEP", "AITIC_ADOPT_ABSTAIN", "AITIC_WIDEN_REFUSAL")},
}
path = os.path.join(SP, "%s_manifest.json" % arm)
with open(path, "w", encoding="utf-8") as f:
    f.write(json.dumps(cfg, ensure_ascii=False, indent=2))
print("[%s] 配置已落盘 → %s" % (arm, os.path.basename(path)))
for k, v in cfg.items():
    if k != "env":
        print("   %-16s %s" % (k, v))
