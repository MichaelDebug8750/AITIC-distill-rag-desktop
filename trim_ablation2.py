# -*- coding: utf-8 -*-
"""
trim_ablation2.py —— 动态上下文裁剪消融（重跑定版）

背景：现有两套数据打架，必须重跑定死：
  《评测报告-动态裁剪小节.md》：900 预算下 相关度 86%(12/14) 覆盖 / 14% 拒答
  后续修正记录：              900 预算下 相关度 64% 覆盖 / 36% 拒答
两者不可能同时成立。本脚本用当前代码重跑，只保留一套。

设计要点（控制变量）：
  · 检索只做一次，两策略共用同一批 docs —— 排除检索波动
  · 直接调用 main.py 的 _pack_relevance / _pack_truncate，不改代码
  · 每个预算档两策略各跑一遍，temperature=0
  · 同一道题在两策略下的上下文若完全相同，标记出来（说明该预算下无区分度）

用法（在 E:\\Ollama_test 下，vectordb 里是对应书的库）：
  & "C:\\Users\\Seifer\\distill\\Scripts\\python.exe" trim_ablation2.py `
      --main E:\\Ollama_test\\data\\main.py --workdir E:\\Ollama_test\\data `
      --eval E:\\Ollama_test\\eval\\eval_by_book --only Microbiology `
      --budgets 500,700,900,1300

输出：屏幕表格 + 消融_动态裁剪_重跑.md
"""
import argparse, glob, io, json, os, sys, importlib.util


def load_main(path):
    spec = importlib.util.spec_from_file_location('dm', path)
    m = importlib.util.module_from_spec(spec)
    sys.argv = ['trim']
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--main', required=True)
    ap.add_argument('--workdir', default='')
    ap.add_argument('--eval', required=True)
    ap.add_argument('--only', default='')
    ap.add_argument('--budgets', default='500,700,900,1300')
    ap.add_argument('--limit', type=int, default=0, help='只跑前 N 道可答题（调试）')
    ap.add_argument('--out', default='消融_动态裁剪_重跑.md')
    a = ap.parse_args()

    wd = os.path.abspath(a.workdir) if a.workdir else os.path.dirname(os.path.abspath(a.main))
    os.chdir(wd)
    mk = os.path.join(wd, '.built_book')
    owner = None
    if os.path.exists(mk):
        try:
            owner = io.open(mk, encoding='utf-8-sig').read().strip()   # -sig：容忍 BOM
        except Exception:
            pass
    print('工作目录: %s ｜ 当前库: %s' % (wd, owner or '未知'))

    # 取题：只用可答题（模糊题/探针不适合测"能不能答出来"）
    rows = []
    for f in sorted(glob.glob(os.path.join(a.eval, '*.jsonl'))):
        rs = [json.loads(l) for l in io.open(f, encoding='utf-8') if l.strip()]
        if not rs:
            continue
        if a.only and a.only.lower() not in rs[0]['book'].lower():
            continue
        rows += [r for r in rs if r['type'] == 'answerable']
    if not rows:
        print('没匹配到可答题，检查 --only'); return
    book = rows[0]['book']
    if owner and owner != book:
        print('!! 库是【%s】建的，题目属于【%s】，结果无效。' % (owner, book))
        if input('   仍要继续？(y/N) ').strip().lower() != 'y':
            return
    if a.limit:
        rows = rows[:a.limit]
    print('题目: %s ｜ 可答题 %d 道\n' % (book, len(rows)))

    M = load_main(os.path.abspath(a.main))
    import chromadb
    col = chromadb.PersistentClient(path=M.DB_PATH).get_collection(M.COLLECTION)
    print('库内块数: %d' % col.count())

    budgets = [int(x) for x in a.budgets.split(',')]

    # ---- 关键：检索只做一次，所有预算/策略共用 ----
    print('检索中（一次性，两策略共用以控制变量）...', flush=True)
    cache = []
    for i, r in enumerate(rows, 1):
        qv = M.embed([r['question']])[0]
        res = col.query(query_embeddings=[qv], n_results=M.TOP_K)
        cache.append((r, res['documents'][0], res['metadatas'][0]))
        if i % 20 == 0:
            print('  %d/%d' % (i, len(rows)), flush=True)

    _selfcheck = {'done': False}

    def ask_with(docs, metas, question, budget, mode):
        if mode == 'rel':
            packed, idx = M._pack_relevance(docs, question, budget)
        else:
            packed, idx = M._pack_truncate(docs, budget)
        ctx = M._labeled_context(packed, idx, metas)
        tags = [M._cite_tag(metas[i]) for i in idx if i < len(metas)]
        uniq = list(dict.fromkeys(tags))[:2]
        tag_example = " or ".join("[%s]" % t for t in uniq) if uniq else "[p.112]"
        out = M._generate(M.LLM_MODEL,
                          M.PROMPT.format(context=ctx, question=question,
                                          tag_example=tag_example),
                          options={"temperature": 0.0, "num_predict": M.NUM_PREDICT})
        # main.py 保证返回对象支持 out["response"] 与 out.get(key, default)，
        # 不要用 isinstance(out, dict) 判断 —— ollama 新版返回的是响应对象而非 dict，
        # 那样会退化成 str(out)（整个对象的 repr），答案里混入元数据。
        ans = M._strip_think(out.get('response', '') or '')
        tok = (out.get('prompt_eval_count', 0) or 0) + (out.get('eval_count', 0) or 0)
        if not _selfcheck['done']:
            _selfcheck['done'] = True
            print('  [自检] 首题答案前 80 字: %r' % ans[:80])
            print('  [自检] token = %d（应 > 0，为 0 说明取数失败）' % tok)
        return ans, tok, ctx

    L = []
    def w(s=''):
        L.append(s); print(s)

    w('# 消融实验 · 动态上下文裁剪（重跑定版）')
    w('')
    w('> 语料：%s（库内 %d 块）｜题目：%d 道可答题' % (book, col.count(), len(rows)))
    w('> **控制变量：检索只做一次，两策略共用同一批检索结果**；`temperature=0`')
    w('')
    w('| 预算 | 策略 | 命中 | 拒答 | 未命中 | token中位 | 上下文与对照相同 |')
    w('|---|---|---|---|---|---|---|')

    table = {}
    for bud in budgets:
        for mode, label in (('trunc', '字符截断（旧）'), ('rel', '相关度整块（新）')):
            hit = ref = miss = same = 0
            toks = []
            for r, docs, metas in cache:
                ans, tok, ctx = ask_with(docs, metas, r['question'], bud, mode)
                if tok:
                    toks.append(tok)
                if M.is_abstain(ans):
                    ref += 1
                elif any(k in ans.lower() for k in r['keywords']):
                    hit += 1
                else:
                    miss += 1
                # 两策略的上下文是否相同（相同 = 该预算下无区分度）
                other = 'rel' if mode == 'trunc' else 'trunc'
                if mode == 'rel':
                    p2, i2 = M._pack_truncate(docs, bud)
                    if M._labeled_context(p2, i2, metas) == ctx:
                        same += 1
            med = sorted(toks)[len(toks) // 2] if toks else 0
            n = len(cache)
            table[(bud, mode)] = (hit, ref, miss, med, same)
            w('| %d | %s | **%d/%d (%.0f%%)** | %d (%.0f%%) | %d | %d | %s |'
              % (bud, label, hit, n, 100.0 * hit / n, ref, 100.0 * ref / n, miss, med,
                 ('%d/%d' % (same, n)) if mode == 'rel' else '—'))
            print('    [%d/%s] 命中%d 拒答%d 未命中%d' % (bud, mode, hit, ref, miss), flush=True)
        w('')

    w('## 结论')
    w('')
    n = len(cache)
    for bud in budgets:
        ht, rt, mt, tt, _ = table[(bud, 'trunc')]
        hr, rr, mr, tr, same = table[(bud, 'rel')]
        d = hr - ht
        if same == n:
            v = '**该预算下两策略上下文完全相同，无区分度**（预算太紧或太松，都退化为同一种打包）'
        elif d > 0:
            v = '相关度整块领先 **+%d 道**（%.0fpp）' % (d, 100.0 * d / n)
        elif d < 0:
            v = '字符截断领先 %d 道（%.0fpp）' % (-d, -100.0 * d / n)
        else:
            v = '两策略持平'
        w('- **预算 %d**：%s；拒答 %d → %d' % (bud, v, rt, rr))
    w('')
    w('> 本次重跑用于替代此前两套互相矛盾的数据（86%/14% 与 64%/36%）。')
    w('> 报告中一律以本表为准。')

    io.open(a.out, 'w', encoding='utf-8').write('\n'.join(L))
    print('\n已写入 %s' % a.out)


if __name__ == '__main__':
    main()
