# -*- coding: utf-8 -*-
"""
calib_gate.py —— 标定 ESCALATE_SIM_GATE 阈值

只做「向量化 + 检索」，不调 LLM，所以很快（几百题约 1–2 分钟）。

原理：
  可答题的最优检索块应该「沾边」（距离小），库外探针应该「不沾边」（距离大）。
  找一个阈值把两者分开：首答拒答后，最优距离大于阈值 → 判为库外，不升配。

用法（在 E:\\Ollama_test 下，且 vectordb 里是这本书的库）：
  & "C:\\Users\\Seifer\\distill\\Scripts\\python.exe" calib_gate.py `
      --main E:\\Ollama_test\\data\\main.py --workdir E:\\Ollama_test\\data `
      --eval E:\\Ollama_test\\eval\\eval_by_book --only Microbiology

建议在 2–3 本不同学科的书上各跑一次，取偏保守（偏大）的建议值。
"""
import argparse, glob, io, json, os, sys, importlib.util


def load_main(path):
    spec = importlib.util.spec_from_file_location('dm', path)
    m = importlib.util.module_from_spec(spec)
    sys.argv = ['calib']            # 防止 main.py 的 argparse 吃到我们的参数
    spec.loader.exec_module(m)
    return m


def q(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0.0
    i = min(len(xs) - 1, max(0, int(round((len(xs) - 1) * p))))
    return xs[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--main', required=True)
    ap.add_argument('--workdir', default='')
    ap.add_argument('--eval', required=True)
    ap.add_argument('--only', default='')
    ap.add_argument('--limit', type=int, default=0, help='每类最多取几题（调试用）')
    a = ap.parse_args()

    wd = os.path.abspath(a.workdir) if a.workdir else os.path.dirname(os.path.abspath(a.main))
    os.chdir(wd)                     # main.py 的 DB_PATH 是相对路径，必须切过去
    print('工作目录: %s' % wd)

    mk = os.path.join(wd, '.built_book')
    owner = io.open(mk, encoding='utf-8').read().strip() if os.path.exists(mk) else None
    print('当前库属于: %s' % (owner or '未知（无 .built_book 标记）'))

    M = load_main(os.path.abspath(a.main))
    import chromadb
    col = chromadb.PersistentClient(path=M.DB_PATH).get_collection(M.COLLECTION)
    print('库内块数: %d\n' % col.count())

    files = sorted(glob.glob(os.path.join(a.eval, '*.jsonl')))
    rows = []
    for f in files:
        rs = [json.loads(l) for l in io.open(f, encoding='utf-8') if l.strip()]
        if not rs:
            continue
        if a.only and a.only.lower() not in rs[0]['book'].lower():
            continue
        rows += rs
    if not rows:
        print('没匹配到题目，检查 --only'); return
    if owner and rows[0]['book'] != owner:
        print('!! 警告：题目属于【%s】，但库是【%s】建的，标定结果无效。'
              % (rows[0]['book'], owner))
        if input('   仍要继续？(y/N) ').strip().lower() != 'y':
            return

    ans, un = [], []
    for i, r in enumerate(rows, 1):
        if a.limit:
            if r['type'] == 'unanswerable' and len(un) >= a.limit:
                continue
            if r['type'] != 'unanswerable' and len(ans) >= a.limit:
                continue
        qv = M.embed([r['question']])[0]
        res = col.query(query_embeddings=[qv], n_results=M.TOP_K)
        d = (res.get('distances') or [[]])[0]
        if not d:
            continue
        (un if r['type'] == 'unanswerable' else ans).append(min(d))
        if i % 25 == 0:
            print('  %d/%d ...' % (i, len(rows)), flush=True)

    if not ans or not un:
        print('样本不足'); return

    print('\n' + '=' * 62)
    print('%-12s %5s %8s %8s %8s %8s' % ('类别', '题数', '最小', '中位', 'P90', '最大'))
    print('%-12s %5d %8.4f %8.4f %8.4f %8.4f'
          % ('可答/模糊', len(ans), min(ans), q(ans, .5), q(ans, .9), max(ans)))
    print('%-12s %5d %8.4f %8.4f %8.4f %8.4f'
          % ('库外探针', len(un), min(un), q(un, .5), q(un, .1), max(un)))
    print('  （库外那行的 P90 列显示的是 P10，即库外题里最"沾边"的那批）')

    # 扫阈值：目标是拦住尽量多的库外升配，同时少误伤可答题
    best = None
    lo, hi = min(ans + un), max(ans + un)
    for k in range(201):
        t = lo + (hi - lo) * k / 200.0
        blocked = sum(1 for x in un if x > t)          # 库外被拦（好）
        hurt = sum(1 for x in ans if x > t)            # 可答被误伤（坏）
        score = blocked / len(un) - 2.0 * hurt / len(ans)   # 误伤惩罚加倍
        if best is None or score > best[0]:
            best = (score, t, blocked, hurt)
    _, t, blocked, hurt = best
    print('\n建议 ESCALATE_SIM_GATE = %.4f' % t)
    print('  → 拦掉库外升配 %d/%d = %.0f%%' % (blocked, len(un), 100.0 * blocked / len(un)))
    print('  → 误伤可答/模糊题 %d/%d = %.1f%%（这些题将不再有升配机会）'
          % (hurt, len(ans), 100.0 * hurt / len(ans)))

    # 更保守的备选：不误伤任何可答题
    safe = max(ans)
    bs = sum(1 for x in un if x > safe)
    print('\n保守备选（零误伤）ESCALATE_SIM_GATE = %.4f' % safe)
    print('  → 拦掉库外升配 %d/%d = %.0f%%，误伤 0%%' % (bs, len(un), 100.0 * bs / len(un)))
    print('\n多本标定时取【偏大】的那个值 —— 宁可少拦，不要误伤可答题。')


if __name__ == '__main__':
    main()
