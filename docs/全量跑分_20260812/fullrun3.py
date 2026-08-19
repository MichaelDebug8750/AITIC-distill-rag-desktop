# -*- coding: utf-8 -*-
"""webui 路径全量（自动发现版）。

和 fullrun2 的区别：不再硬编码书名。启动时把题集里的书名和当前已建的知识库
做匹配，**有库的书才跑**。这样每建一本新库，覆盖面自动扩大，不用改脚本。

匹配用文件名主干做规范化比对，不做模糊猜测——猜错会把 A 书的题打到 B 库上，
得出的准确率全是假的，比不跑更糟。匹配不上的书直接跳过并列出来。

用法：fullrun3.py <port> <tag> [auto|on|off] [resume.jsonl]

第三个参数显式控制每次请求的 hybrid 字段；省略时沿用服务默认。运行配置同时
写入 ``<tag>_manifest.json``，避免只靠文件名猜测实验臂实际开关。
若结果只保存成了另一个文件名，第四个参数会先复制该备份再续跑；源文件保留不动。
"""
import io
import hashlib
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request

from eval_compare import row_key

if len(sys.argv) < 3 or len(sys.argv) > 5:
    raise SystemExit("用法：fullrun3.py <port> <tag> [auto|on|off] [resume.jsonl]")
PORT, TAG = sys.argv[1], sys.argv[2]
if not re.fullmatch(r"\d{1,5}", PORT) or not 1 <= int(PORT) <= 65535:
    raise SystemExit("port 必须是 1–65535 的整数")
if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", TAG):
    raise SystemExit("tag 只能含字母、数字、下划线和连字符，且不能以符号开头")
HYBRID_REQUEST = None
if len(sys.argv) > 3:
    value = str(sys.argv[3]).strip().lower()
    if value not in ("auto", "on", "off"):
        raise SystemExit("hybrid 必须是 auto/on/off")
    HYBRID_REQUEST = None if value == "auto" else value == "on"
RESUME_FROM = os.path.abspath(sys.argv[4]) if len(sys.argv) > 4 else None
B = "http://127.0.0.1:%s" % PORT
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
EVAL = os.path.join(PROJECT_ROOT, "eval", "eval_ALL.jsonl")
ROWS_PATH = os.path.join(HERE, "%s_rows.jsonl" % TAG)
MANIFEST_PATH = os.path.join(HERE, "%s_manifest.json" % TAG)
NO_REF = "[NO REFERENCE FOUND]"
CITE = re.compile(r"\[[^\]]+\]")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fingerprints():
    paths = [
        EVAL,
        os.path.join(PROJECT_ROOT, "code", "webui.py"),
        os.path.join(PROJECT_ROOT, "code", "webui_index.html"),
        os.path.join(PROJECT_ROOT, "code", "test_pipeline.py"),
        os.path.join(PROJECT_ROOT, "code", "main.py"),
        __file__,
    ]
    return {os.path.relpath(path, PROJECT_ROOT): sha256(path) for path in paths}


def write_manifest(payload):
    temp_manifest = "%s.%d.tmp" % (MANIFEST_PATH, os.getpid())
    with io.open(temp_manifest, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp_manifest, MANIFEST_PATH)


RESUME_MANIFEST = None
if RESUME_FROM:
    if os.path.exists(ROWS_PATH):
        raise SystemExit("目标结果文件已存在，不能再指定 resume.jsonl：%s" % ROWS_PATH)
    if not os.path.isfile(RESUME_FROM):
        raise SystemExit("续跑备份不存在：%s" % RESUME_FROM)
    if os.path.normcase(RESUME_FROM) == os.path.normcase(os.path.abspath(ROWS_PATH)):
        raise SystemExit("续跑备份不能与目标结果文件相同")
    if not RESUME_FROM.endswith("_rows.jsonl"):
        raise SystemExit("续跑备份必须是 fullrun 生成的 *_rows.jsonl")
    RESUME_MANIFEST = RESUME_FROM[:-len("_rows.jsonl")] + "_manifest.json"
    if not os.path.isfile(RESUME_MANIFEST):
        raise SystemExit("续跑备份缺少同名 manifest，无法证明构建身份：%s" % RESUME_MANIFEST)


def norm(name):
    """书名规范化：去扩展名、去标点、压空白、转小写。

    题集里是 'Anatomy_and_Physiology_2e.pdf'，库名可能是 'Anatomy_and_Physiology_2e'
    或带破折号的变体。只做确定性的字符归一，不做编辑距离之类的模糊匹配。
    """
    s = os.path.splitext(str(name or ""))[0]
    s = re.sub(r"[^\w\u4e00-\u9fff]+", "", s)
    return s.lower()


def libraries():
    with urllib.request.urlopen(B + "/api/libraries", timeout=180) as r:
        d = json.loads(r.read().decode("utf-8"))
    return d.get("libraries") or d.get("items") or d


def ask(question, lib, timeout=900, attempts=3):
    body = {"question": question, "libraries": [lib], "mode": "auto",
            "style": "standard", "extend": False, "history": []}
    if HYBRID_REQUEST is not None:
        body["hybrid"] = HYBRID_REQUEST
    rq = urllib.request.Request(B + "/api/ask", data=json.dumps(body).encode("utf-8"),
                                headers={"Content-Type": "application/json"})
    last = (0, {"error": "request not attempted"})
    for attempt in range(max(1, int(attempts))):
        try:
            with urllib.request.urlopen(rq, timeout=timeout) as r:
                return r.status, json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", "replace")
            try:
                payload = json.loads(raw or "{}")
            except ValueError:
                payload = {"error": raw[:300] or "HTTP %d" % exc.code}
            last = (exc.code, payload)
            if exc.code < 500 or attempt + 1 >= attempts:
                return last
        except Exception as exc:
            last = (0, {"error": "%s: %s" % (type(exc).__name__, str(exc)[:120])})
            if attempt + 1 >= attempts:
                return last
        time.sleep(min(2 ** attempt, 4))
    return last


try:
    with urllib.request.urlopen(B + "/api/status", timeout=180) as response:
        service_status = json.loads(response.read().decode("utf-8"))
except Exception as exc:
    service_status = {"status_error": "%s: %s" % (type(exc).__name__, str(exc)[:160])}
service_config = {key: service_status.get(key) for key in (
    "llm_model", "embed_model", "hybrid_default", "evidence_floor",
    "style_gate_max", "widen_refusal", "keyword_df_ratio", "db_path", "runtime")
    if key in service_status}
libs = [x for x in libraries() if str(x.get("status") or "ready") == "ready"]
by_norm = {}
for x in libs:
    for key in (x.get("source"), x.get("name")):
        if key:
            by_norm.setdefault(norm(key), x.get("id"))

cases, unmatched = [], set()
for line in io.open(EVAL, encoding="utf-8"):
    row = json.loads(line)
    book = row.get("book") or ""
    lib = by_norm.get(norm(book))
    if lib:
        # 结果身份保留完整书名；28 字截断只适合终端展示，不能拿来做复合键。
        cases.append((row, lib, os.path.splitext(book)[0]))
    else:
        unmatched.add(book)

used_library_ids = {case[1] for case in cases}


def snapshot_for(all_libraries):
    snapshot = [{key: item.get(key) for key in
                 ("id", "name", "source", "chunks", "built_at", "status")}
                for item in all_libraries if item.get("id") in used_library_ids]
    return sorted(snapshot, key=lambda item: item.get("id") or "")


library_snapshot = snapshot_for(libs)
if len(library_snapshot) != len(used_library_ids):
    raise SystemExit("无法为全部题集知识库建立构建快照")
identity = {
    "hybrid_request": HYBRID_REQUEST,
    "service_config": service_config,
    "fingerprints": source_fingerprints(),
    "libraries": library_snapshot,
}
manifest = dict(identity, tag=TAG, port=PORT,
                recorded_at=time.strftime("%Y-%m-%d %H:%M:%S"))


def validate_identity(prior, label):
    if prior.get("hybrid_request") != identity["hybrid_request"]:
        raise SystemExit("%s的 hybrid_request 不一致，拒绝混合实验配置" % label)
    if (prior.get("service_config") or {}) != identity["service_config"]:
        raise SystemExit("%s的服务配置不一致，拒绝把不同构建混进结果文件" % label)
    if prior.get("fingerprints") != identity["fingerprints"]:
        raise SystemExit("%s的题集/核心源码指纹变化，拒绝混合构建" % label)
    if prior.get("libraries") != identity["libraries"]:
        raise SystemExit("%s的知识库构建快照变化，拒绝混合向量库" % label)


if os.path.exists(ROWS_PATH) and not os.path.exists(MANIFEST_PATH):
    raise SystemExit("结果文件缺少 manifest，无法证明构建身份；请换新 tag 或提供带清单的续跑备份")
if RESUME_FROM:
    with io.open(RESUME_MANIFEST, encoding="utf-8") as handle:
        resume_manifest = json.load(handle)
    validate_identity(resume_manifest, "续跑备份")
    shutil.copyfile(RESUME_FROM, ROWS_PATH)
    print("[%s] 已验证清单并复制续跑起点：%s（源文件保留）" %
          (TAG, RESUME_FROM), flush=True)
if os.path.exists(MANIFEST_PATH):
    with io.open(MANIFEST_PATH, encoding="utf-8") as handle:
        prior_manifest = json.load(handle)
    validate_identity(prior_manifest, "现有 manifest")
    manifest = prior_manifest
else:
    write_manifest(manifest)

print("[%s] 已建库 %d 个；题集 %d 本书中 %d 本有库，可跑 %d 题"
      % (TAG, len(libs), len(unmatched) + len({c[2] for c in cases}),
         len({c[2] for c in cases}), len(cases)), flush=True)
if unmatched:
    print("  无库跳过 %d 本：%s" % (len(unmatched), "、".join(sorted(unmatched)[:6]) + " …"), flush=True)

# 不可答题先跑（多在检索闸门就返回），可答题按书轮转 —— 中断时样本仍均衡
def interleave(items):
    buckets, order = {}, []
    for c in items:
        if c[2] not in buckets:
            buckets[c[2]] = []; order.append(c[2])
        buckets[c[2]].append(c)
    out, i = [], 0
    while any(buckets[k][i:] for k in order):
        for k in order:
            if i < len(buckets[k]):
                out.append(buckets[k][i])
        i += 1
    return out

cases = ([c for c in cases if c[0].get("expect") == "abstain"]
         + interleave([c for c in cases if c[0].get("expect") != "abstain"]))

done = set()
resume_rows = []
needs_compaction = False
if os.path.exists(ROWS_PATH):
    for line in io.open(ROWS_PATH, encoding="utf-8"):
        if line.strip():
            try:
                saved = json.loads(line)
            except (ValueError, TypeError):
                needs_compaction = True
                continue
            # 瞬时网络/服务失败不是评测结果。恢复运行时必须重试，不能永久写进分母。
            status = saved.get("status")
            if (saved.get("outcome") != "请求失败" and isinstance(status, int)
                    and 200 <= status < 300):
                key = row_key(saved)
                if key in done:
                    raise RuntimeError("结果文件存在重复复合键：%r" % (key,))
                done.add(key)
                resume_rows.append(saved)
            else:
                needs_compaction = True
# 重试前先原子移除失败/损坏行，否则成功结果 append 后会与失败行形成重复复合键。
if needs_compaction:
    compact_path = ROWS_PATH + ".resume.tmp"
    with io.open(compact_path, "w", encoding="utf-8") as handle:
        for saved in resume_rows:
            handle.write(json.dumps(saved, ensure_ascii=False) + "\n")
    os.replace(compact_path, ROWS_PATH)
todo = [c for c in cases
        if row_key({"book": c[2], "question": c[0]["question"]}) not in done]
print("[%s] 已完成 %d，本次待跑 %d" % (TAG, len(done), len(todo)), flush=True)

t_start = time.time()
for i, (row, lib, key) in enumerate(todo, 1):
    t0 = time.time()
    st, d = ask(row["question"], lib)
    elapsed = time.time() - t0
    answer = str(d.get("answer") or "").strip()
    abstained = bool(d.get("abstained"))
    agent = d.get("agent") or {}
    audit = agent.get("support_audit") or {}
    kws = [str(k).lower() for k in (row.get("keywords") or [])]
    body_txt = CITE.sub("", answer).lower()
    hit = any(k in body_txt for k in kws) if kws else None

    request_failed = not isinstance(st, int) or not (200 <= st < 300)
    if request_failed:
        outcome = "请求失败"
    elif row.get("expect") == "abstain":
        outcome = "拒答正确" if (abstained and answer == NO_REF) else "编造"
    elif abstained:
        outcome = "过度拒答"
    elif hit is None:
        outcome = "未判定"
    else:
        outcome = "命中" if hit else "未命中"

    rec = {"question": row["question"], "book": key, "library_id": lib,
           "type": row.get("type"),
           "expect": row.get("expect"), "status": st, "outcome": outcome,
           "abstained": abstained, "rounds": agent.get("rounds"),
           "cite_ok": bool((d.get("cite_check") or {}).get("ok")),
           "confidence": (agent.get("confidence") or {}).get("level"),
           "pruned": audit.get("pruned"), "orphaned": audit.get("orphaned"),
           "unknown": audit.get("unknown"), "stop_reason": agent.get("stop_reason"),
           "elapsed": round(elapsed, 1), "tokens": d.get("tokens"),
           # 完整答案是人工复核编造/引用的证据；截成 800 字会把关键尾句永久丢掉。
           "answer": answer, "error": d.get("error")}
    with io.open(ROWS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if i % 20 == 0 or i == len(todo):
        spent = (time.time() - t_start) / 60
        print("%5d/%-5d 已用 %.0f 分钟，预计还需 %.0f 分钟"
              % (i, len(todo), spent, spent / i * (len(todo) - i)), flush=True)

print("[%s] 本次结束，用时 %.1f 分钟" % (TAG, (time.time() - t_start) / 60), flush=True)
end_fingerprints = source_fingerprints()
if end_fingerprints != manifest["fingerprints"]:
    raise RuntimeError("运行期间题集或核心源码发生变化，结果不得作为同一构建基线")
end_libraries = snapshot_for(
    [item for item in libraries() if str(item.get("status") or "ready") == "ready"])
if end_libraries != manifest.get("libraries"):
    raise RuntimeError("运行期间知识库构建快照发生变化，结果不得作为同一构建基线")
with io.open(ROWS_PATH, encoding="utf-8") as handle:
    completed_rows = sum(1 for line in handle if line.strip())
manifest.update({"completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                 "completed_rows": completed_rows,
                 "rows_sha256": sha256(ROWS_PATH),
                 "end_fingerprints": end_fingerprints,
                 "end_libraries": end_libraries})
write_manifest(manifest)
