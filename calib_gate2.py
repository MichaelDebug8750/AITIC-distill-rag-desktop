# -*- coding: utf-8 -*-
"""
calib_gate2.py —— 精确版闸门标定

比 calib_gate.py 强在哪：
  v1 只看「可答题 vs 库外题」的距离分布，但闸门只在【首答已拒答】时才起作用，
  所以真正该分开的是「升配救回的题」和「升配白跑的题」——这两批都是首答拒答的。
  本脚本直接读 v1 基线的逐题结果，拿真实的升配记录来标定，目标明确：
      保住所有救回的题，拦掉尽可能多的浪费。

前提：
  · vectordb 里必须是这本书的库（脚本会校验 .built_book）
  · 有该书的 v1 基线结果（eval_results_v1_baseline\\ 或 eval_results\\）

用法：
  & "C:\\Users\\Seifer\\distill\\Scripts\\python.exe" calib_gate2.py `
      --main E:\\Ollama_test\\data\\main.py --workdir E:\\Ollama_test\\data `
      --results E:\\Ollama_test\\eval_results_v1_baseline --only Microbiology
"""
import argparse, glob, io, json, os, sys, importlib.util


def load_main(path):
    spec = importlib.util.spec_from_file_location('dm', path)
    m = importlib.util.module_from_spec(spec)
    sys.argv = ['calib']
    spec.loader.exec_module(m)
    return m


def stat(xs):
    if not xs:
        return '—'
    s = sorted(xs)
    return 'n=%d 最小%.4f 中位%.4f 最大%.4f' % (len(s), s[0], s[len(s) // 2], s[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--main', required=True)
    ap.add_argument('--workdir', default='')
    ap.add_argument('--results', required=True, help='v1 基线结果目录')
    ap.add_argument('--only', default='', help='书名关键字')
    a = ap.parse_args()

    wd = os.path.abspath(a.workdir) if a.workdir else os.path.dirname(os.path.abspath(a.main))
    os.chdir(wd)
    mk = os.path.join(wd, '.built_book')
    owner = io.open(mk, encoding='utf-8').read().strip() if os.path.exists(mk) else None
    print('工作目录: %s' % wd)
    print('当前库属于: %s' % (owner or '未知'))

    rows = []
    for f in sorted(glob.glob(os.path.join(a.results, '*.jsonl'))):
        rs = [json.loads(l) for l in io.open(f, encoding='utf-8') if l.strip()]
        if not rs:
            continue
        if a.only and a.only.lower() not in rs[0]['book'].lower():
            continue
        rows += rs
    if not rows:
        print('没找到该书的 v1 结果，检查 --results / --only'); return
    book = rows[0]['book']
    print('标定书目: %s（v1 共 %d 题）' % (book, len(rows)))
    if owner and owner != book:
        print('!! 库是【%s】建的，与标定书目不符，结果无效。' % owner)
        if input('   仍要继续？(y/N) ').strip().lower() != 'y':
            return

    esc = [r for r in rows if r.get('escalated')]
    if not esc:
        print('该书 v1 没有触发过升配，无法标定。换一本。'); return
    print('触发过升配 %d 道\n' % len(esc))

    M = load_main(os.path.abspath(a.main))
    import chromadb
    col = chromadb.PersistentClient(path=M.DB_PATH).get_collection(M.COLLECTION)
    print('库内块数: %d，开始算距离 ...' % col.count())

    saved, wasted = [], []
    for i, r in enumerate(esc, 1):
        qv = M.embed([r['question']])[0]
        res = col.query(query_embeddings=[qv], n_results=M.TOP_K)
        d = (res.get('distances') or [[]])[0]
        if not d:
            continue
        best = min(d)
        # 救回 = 非库外题 且 升配后答对了（不升配的话这题就是拒答）
        if r['type'] != 'unanswerable' and r['ok']:
            saved.append(best)
        else:
            wasted.append(best)
        if i % 20 == 0:
            print('  %d/%d ...' % (i, len(esc)), flush=True)

    print('\n' + '=' * 64)
    print('升配救回的题（闸门绝不能拦）: %s' % stat(saved))
    print('升配白跑的题（闸门应该拦掉）: %s' % stat(wasted))

    if not saved:
        print('\n该书升配一道也没救回 —— 说明对这本书升配纯属浪费。')
        print('单本无法定出上界，请换一本有救回记录的书再标。')
        if wasted:
            print('参考：把阈值设在 %.4f 以下即可拦掉全部浪费。' % min(wasted))
        return

    gate = max(saved)
    blocked = sum(1 for x in wasted if x > gate)
    print('\n【零误伤阈值】ESCALATE_SIM_GATE = %.4f' % gate)
    print('  → 保住全部 %d 道救回题' % len(saved))
    print('  → 拦掉浪费 %d/%d = %.0f%%' % (blocked, len(wasted), 100.0 * blocked / max(1, len(wasted))))

    # 再给几档更激进的，标注代价
    print('\n更激进的档位（会牺牲部分救回题，供权衡）:')
    s = sorted(saved, reverse=True)
    for k in (1, 2, 3):
        if len(s) > k:
            g = s[k]
            b = sum(1 for x in wasted if x > g)
            print('  阈值 %.4f → 拦掉浪费 %.0f%%，牺牲救回题 %d/%d'
                  % (g, 100.0 * b / max(1, len(wasted)), k, len(saved)))

    print('\n多本标定时取【最大】的零误伤阈值 —— 保守优先。')


if __name__ == '__main__':
    main()
