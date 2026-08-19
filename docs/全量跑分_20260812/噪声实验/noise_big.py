# -*- coding: utf-8 -*-
"""有统计效力的噪声底测量：同题连跑 N 次，看作答/拒答是否翻转。

题目身份必须是 ``(book, question)``。正式 1007 题中有跨书复用的同名问题，
只按问题文本映射会把题发给另一本文库；断点续跑也会误把另一书的同题当成已完成。

为什么要重做：此前用 23 题测过三次，得到 9% / 17% / 13%。三个数在 n=23 下
完全分不开（2 条 vs 3 条 vs 4 条），据此做的任何取舍都是在噪声里打转——
今晚已经因此白试了两个改动。

样本量：100 题 × 3 次。若真实翻转率约 9%，n=100 的标准误约 2.9%，
可以分辨 9% 与 2%，但仍分辨不了 9% 与 6%。这是这次能达到的精度上限，
结论里必须照实说，不能把 13%→9% 这种差异当成改善。

用法：noise_big.py <tag> <题数> [重复次数] [基线 rows.jsonl]
"""
import hashlib
import io
import json
import os
import re
import sys
import time
import urllib.request

SP = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SP, "..", "..", ".."))
B = "http://127.0.0.1:" + (os.environ.get("NOISE_PORT") or "8011")
if len(sys.argv) < 2:
    raise SystemExit("用法：noise_big.py <tag> <题数> [重复次数] [基线 rows.jsonl]")
TAG = sys.argv[1]
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", TAG):
    raise SystemExit("tag 只能含字母、数字、下划线和连字符，且不能以符号开头")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
REPS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
if not 2 <= N <= 1000 or not 2 <= REPS <= 20:
    raise SystemExit("题数必须在 2–1000，重复次数必须在 2–20")
BASELINE_ARG = sys.argv[4] if len(sys.argv) > 4 else "after_rows.jsonl"
BASELINE = (BASELINE_ARG if os.path.isabs(BASELINE_ARG)
            else os.path.abspath(os.path.join(SP, BASELINE_ARG)))
OUT = os.path.join(SP, "noisebig_%s.jsonl" % TAG)
MANIFEST = os.path.join(SP, "noisebig_%s_manifest.json" % TAG)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_manifest(payload):
    temp = MANIFEST + ".%d.tmp" % os.getpid()
    with io.open(temp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, MANIFEST)


def norm(n):
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", os.path.splitext(str(n or ""))[0]).lower()


if not os.path.isfile(BASELINE):
    raise SystemExit("基线结果不存在：%s" % BASELINE)
status = json.loads(urllib.request.urlopen(B + "/api/status", timeout=180).read().decode("utf-8"))
service_config = {key: status.get(key) for key in (
    "llm_model", "embed_model", "hybrid_default", "evidence_floor",
    "style_gate_max", "widen_refusal", "keyword_df_ratio", "db_path", "runtime")
    if key in status}
source_paths = [os.path.join(PROJECT_ROOT, "code", "webui.py"),
                os.path.join(PROJECT_ROOT, "code", "main.py"), __file__]
libs = json.loads(urllib.request.urlopen(B + "/api/libraries", timeout=180).read().decode("utf-8"))
libs = libs.get("libraries") or libs.get("items") or libs
lid, library_size = {}, {}
for x in libs:
    if x.get("id"):
        library_size[x["id"]] = int(x.get("chunks") or 0)
    for k in (x.get("source"), x.get("name")):
        if k:
            lid.setdefault(norm(k), x.get("id"))


def resolve_library_id(book):
    """fullrun 历史行把书名截到 28 字；只允许精确或唯一前缀匹配。"""
    key = norm(book)
    if key in lid:
        return lid[key]
    candidates = {library_id for name, library_id in lid.items()
                  if key and name.startswith(key) and library_id}
    return next(iter(candidates)) if len(candidates) == 1 else None

rows = [json.loads(l) for l in io.open(BASELINE, encoding="utf-8")
        if l.strip()]

# 分层抽样：可答/不可答各半，且跨库规模铺开。
# 只抽原本命中的可答题——它们最能暴露"本来答得出、重跑却拒答"的翻转；
# 不可答题则暴露反向翻转。只抽一类会看不见另一半。
pool = []
for r in rows:
    # fullrun 行里的 book（即使是历史 28 字截断名）与 question 共同组成身份；
    # 不再用 question 单键回查题集，避免跨书同题绑定到先出现的教材。
    library_id = resolve_library_id(r.get("book") or "")
    if not library_id:
        continue
    if r["outcome"] in ("命中", "拒答正确"):
        pool.append((r, library_id, library_size.get(library_id, 0),
                     r["expect"] == "abstain"))

ans = [p for p in pool if not p[3]]
una = [p for p in pool if p[3]]
half = N // 2
step_a = max(1, len(ans) // half) if half else 1
step_u = max(1, len(una) // half) if half else 1
sample = ans[::step_a][:half] + una[::step_u][:N - half]
if len(sample) != N:
    raise SystemExit("可用分层样本不足：要求 %d，实际 %d" % (N, len(sample)))
used_library_ids = {item[1] for item in sample}
library_snapshot = sorted(({
    key: library.get(key) for key in
    ("id", "name", "source", "chunks", "built_at", "status")
} for library in libs if library.get("id") in used_library_ids), key=lambda item: item.get("id") or "")
if len(library_snapshot) != len(used_library_ids):
    raise SystemExit("无法为全部抽样知识库建立构建快照")
identity = {
    "tag": TAG, "base_url": B, "sample_size": N, "repeats": REPS,
    "baseline": BASELINE, "baseline_sha256": sha256(BASELINE),
    "service_config": service_config, "libraries": library_snapshot,
    "fingerprints": {os.path.relpath(path, PROJECT_ROOT): sha256(path)
                     for path in source_paths},
}
if os.path.exists(OUT) and not os.path.exists(MANIFEST):
    raise SystemExit("结果已存在但缺少 manifest；不能证明同构建续跑，请换新 tag：%s" % OUT)
if os.path.exists(MANIFEST):
    prior = json.load(io.open(MANIFEST, encoding="utf-8"))
    prior_identity = {key: prior.get(key) for key in identity}
    if prior_identity != identity:
        raise SystemExit("续跑的基线、知识库、服务配置或源码指纹变化，拒绝混入同一噪声实验")
    manifest = prior
else:
    manifest = dict(identity, started_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    write_manifest(manifest)
print("[%s] 抽样 %d 题（可答 %d / 不可答 %d），每题跑 %d 次 = %d 次调用"
      % (TAG, len(sample), sum(1 for s in sample if not s[3]),
         sum(1 for s in sample if s[3]), REPS, len(sample) * REPS), flush=True)


def ask(q, lib):
    body = {"question": q, "libraries": [lib], "mode": "auto",
            "style": "standard", "extend": False, "history": []}
    rq = urllib.request.Request(B + "/api/ask", data=json.dumps(body).encode("utf-8"),
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(rq, timeout=900) as r:
            d = json.loads(r.read().decode("utf-8"))
        return bool(d.get("abstained")), (d.get("agent") or {}).get("support_audit") or {}
    except Exception:
        return None, {}


done = set()
legacy_done_questions = set()
if os.path.exists(OUT):
    for line in io.open(OUT, encoding="utf-8"):
        if line.strip():
            try:
                previous = json.loads(line)
                if previous.get("book"):
                    done.add((previous["book"], previous["question"]))
                else:
                    # 兼容旧结果：旧文件没有 book，无法恢复真实复合身份；
                    # 保持旧行为只按问题跳过，避免续跑时重复追加一整批。
                    legacy_done_questions.add(previous["question"])
            except Exception:
                pass

t0 = time.time()
todo = [s for s in sample
        if (s[0].get("book"), s[0]["question"]) not in done
        and s[0]["question"] not in legacy_done_questions]
if not os.path.exists(OUT):
    io.open(OUT, "a", encoding="utf-8").close()
for i, (r, lib, size, is_una) in enumerate(todo, 1):
    outs = []
    for _ in range(REPS):
        ab, audit = ask(r["question"], lib)
        if ab is None:
            break
        outs.append(ab)
    if len(outs) < REPS:
        continue
    with io.open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps({"book": r.get("book"), "question": r["question"], "chunks": size,
                            "unanswerable": is_una, "outs": outs,
                            "flip": len(set(outs)) > 1}, ensure_ascii=False) + "\n")
    if i % 10 == 0 or i == len(todo):
        spent = (time.time() - t0) / 60
        print("  %d/%d 已用 %.0f 分钟，预计还需 %.0f 分钟"
              % (i, len(todo), spent, spent / i * (len(todo) - i)), flush=True)

recs = [json.loads(l) for l in io.open(OUT, encoding="utf-8") if l.strip()]
record_keys = [(r.get("book"), r.get("question")) for r in recs]
if len(record_keys) != len(set(record_keys)):
    raise SystemExit("噪声结果存在重复 (book, question)，拒绝汇总损坏的续跑文件")
manifest.update({
    "completed_records": len(recs),
    "result_sha256": sha256(OUT) if os.path.isfile(OUT) else "",
    "completed_at": time.strftime("%Y-%m-%d %H:%M:%S") if len(recs) == N else "",
})
write_manifest(manifest)
flips = sum(1 for x in recs if x["flip"])
n = len(recs)
se = (flips / n * (1 - flips / n) / n) ** 0.5 if n else 0
print()
print("[%s] 样本 %d 题，翻转 %d 条 = %.1f%%  (±%.1f%% 标准误，95%%区间约 %.1f%%–%.1f%%)"
      % (TAG, n, flips, 100.0 * flips / n if n else 0, 100 * se,
         100 * max(0, flips / n - 1.96 * se) if n else 0,
         100 * min(1, flips / n + 1.96 * se) if n else 0))
for label, sel in (("可答题", lambda x: not x["unanswerable"]),
                   ("不可答题", lambda x: x["unanswerable"]),
                   ("大库≥4000", lambda x: x["chunks"] >= 4000),
                   ("小中库<4000", lambda x: 0 < x["chunks"] < 4000)):
    sub = [x for x in recs if sel(x)]
    if sub:
        print("   %-12s %3d 题，翻转 %2d = %.1f%%"
              % (label, len(sub), sum(1 for x in sub if x["flip"]),
                 100.0 * sum(1 for x in sub if x["flip"]) / len(sub)))
