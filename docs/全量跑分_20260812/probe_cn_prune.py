# -*- coding: utf-8 -*-
"""中文过度拒答：是模型直接拒答，还是"答了但被逐句裁剪裁光了"。

诊断已知：10/11 道的证据就在 top-8 里、闸门没拦，说明模型是拿着证据拒的。
但"拿着证据拒"有两种完全不同的成因：

  A. 模型自己输出了 [NO REFERENCE FOUND]        → 病在提示词/生成
  B. 模型答了，但逐句核验把所有结论都判为无据裁光 → 病在接地率算法（中文分词）

两者的修法完全相反，必须分开。看 support_audit 里的 pruned/unknown 即可判定。
用法：probe_cn_prune.py [port]
"""
import io
import json
import os
import sys
import time
import urllib.request

SP = os.path.dirname(os.path.abspath(__file__))
B = "http://127.0.0.1:" + (sys.argv[1] if len(sys.argv) > 1 else "8011")

QS = [
    "舍克勒是什么？它的重量是多少？",
    "银行的英文 bank 是怎么来的？",
    "世界最早的纸币是什么？出现在哪里？",
    "谁被称作“复式记账法之父”？他写了什么书？",
    "汇票和支票有什么区别？",
]

libs = json.loads(urllib.request.urlopen(B + "/api/libraries", timeout=180).read().decode("utf-8"))
libs = libs.get("libraries") or libs.get("items") or libs
lid = None
for x in libs:
    if "简明世界经济史" in str(x.get("name") or ""):
        lid = x.get("id")
if not lid:
    raise SystemExit("服务里没有这个库")

for q in QS:
    body = {"question": q, "libraries": [lid], "mode": "auto",
            "style": "standard", "extend": False, "history": []}
    rq = urllib.request.Request(B + "/api/ask", data=json.dumps(body).encode("utf-8"),
                                headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(rq, timeout=900) as resp:
            d = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print("%-26s 请求失败 %r" % (q[:24], exc)); continue
    ag = d.get("agent") or {}
    au = ag.get("support_audit") or {}
    ans = (d.get("answer") or "").strip()
    print("=" * 72)
    print("Q: %s   (%.1fs, %s 轮)" % (q, time.time() - t0, ag.get("rounds")))
    print("  拒答=%s  引用校验=%s  可信度=%s"
          % (d.get("abstained"), (d.get("cite_check") or {}).get("ok"),
             (ag.get("confidence") or {}).get("level")))
    print("  裁剪=%s 悬空=%s 未判定=%s   ← 裁剪>0 且最终拒答 = 被裁光"
          % (au.get("pruned"), au.get("orphaned"), au.get("unknown")))
    print("  停止原因: %s" % str(ag.get("stop_reason"))[:70])
    print("  答案: %s" % ans[:110].replace("\n", " "))
    sup = (d.get("supplement") or {}).get("text") or ""
    if sup:
        print("  上段完整解答 %d 字（说明模型其实答得出来）: %s"
              % (len(sup), sup[:90].replace("\n", " ")))
