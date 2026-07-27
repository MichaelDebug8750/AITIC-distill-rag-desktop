# -*- coding: utf-8 -*-
"""
run_eval_batch.py —— 批量建库 + 跑三段式题集 + 出报告
已按 main.py 实际接口对接，无需再改任何东西。

对接要点（都是读你 main.py 确认过的）：
  · PDF  : build --pdf PATH --max-pages 999999 --no-vl
           ！！max-pages 默认只有 120，不覆盖的话第 120 页之后的题全部假 MISS
  · EPUB : build --epub PATH
           ！！build_epub 是 append 不删库，所以每次建 EPUB 前先删 ./vectordb
  · 提问 : ask "问题"
  · 拒答 : [NO REFERENCE FOUND]（与 main.py 的 is_abstain 同口径）
  · 顺带采集 tokens 和动态升配次数，可直接写进评测报告

用法：
  # 先小样验通（Microbiology 前 10 题）
  & "C:\\Users\\Seifer\\distill\\Scripts\\python.exe" run_eval_batch.py `
      --books E:\\Ollama_test\\books --eval E:\\Ollama_test\\eval\\eval_by_book `
      --only Microbiology --limit 10

  # 跑整本
  ... --only Microbiology

  # 只跑术语表子集（主指标建议用这个）
  ... --only Microbiology --source glossary

  # 全量（很慢，挂着过夜；断了加 --resume 接着跑）
  ... --books E:\\Ollama_test\\books --eval E:\\Ollama_test\\eval\\eval_by_book --resume
"""
import argparse, glob, io, json, os, re, shutil, subprocess, sys, time
from collections import Counter

PY = sys.executable
MAIN = 'main.py'          # 由 --main 覆盖
WORKDIR = '.'             # 子进程工作目录 = main.py 所在目录
DB_PATH = './vectordb'    # main.py 里是相对路径，故随 WORKDIR 走

BUILD_TIMEOUT = 14400      # 单本建库上限 4 小时（A&P 1347 页会很久）
ASK_TIMEOUT = 300

# 与 main.py is_abstain 同口径
ABSTAIN = re.compile(r'no reference found|\[[^\]]*\bno\b[^\]]*\breferences?\b', re.I)


# Windows 上子进程 print() 默认走 cp936，管道读出来会乱码，
# 导致切不出 [来源] 分隔行。强制子进程用 UTF-8。
ENV = dict(os.environ)
ENV['PYTHONIOENCODING'] = 'utf-8'
ENV['PYTHONUTF8'] = '1'


def run(cmd, timeout):
    try:
        p = subprocess.run(cmd, capture_output=True, timeout=timeout,
                           encoding='utf-8', errors='replace', env=ENV, cwd=WORKDIR)
        return (p.stdout or '') + (p.stderr or ''), p.returncode
    except subprocess.TimeoutExpired:
        return '__TIMEOUT__', -9


def parse_ask(raw):
    """从 ask 的输出里切出答案正文、tokens、是否动态升配"""
    if raw == '__TIMEOUT__':
        return '', 0, False, True
    tok = 0
    m = re.search(r'tokens:\s*(\d+)', raw)
    if m:
        tok = int(m.group(1))
    esc = '动态升配' in raw
    # 正常情况按 [来源] 切；万一编码仍有问题，退回按 tokens: 那行切
    i = raw.find('\n[来源]')
    if i < 0:
        m2 = re.search(r'\n\[[^\]]{0,12}\][^\n]*tokens:', raw)
        i = m2.start() if m2 else -1
    ans = raw[:i] if i > 0 else raw
    return ans.strip(), tok, esc, False


def is_abstain(ans):
    return (not ans.strip()) or bool(ABSTAIN.search(ans))


def _overlap(ans, ev, question=''):
    """答案里【超出问句】的部分与原文证据句的重合度。

    必须扣掉问句已有的词：fuzzy_desc 的问句本身就是证据句改的，
    模型只要复读问句重合度就虚高，会把"没答"误判成"答对了没点名"。
    """
    a = ans.lower()
    q = set(re.findall(r'[a-z]{5,}', question.lower()))
    w = [x for x in re.findall(r'[a-z]{5,}', (ev or '').lower()) if x not in q]
    if len(w) < 3:            # 证据句几乎全被问句覆盖 → 无法判别，不给宽松分
        return 0.0
    return sum(1 for x in w if x in a) / len(w)


def _page_match(row, ans, tol=2):
    """答案里的引用页是否落在证据页附近。
    只对非 glossary 来源有效 —— glossary 的 page 记的是术语表页不是正文页。
    """
    if row.get('source') == 'glossary' or not row.get('page'):
        return False
    cited = [int(x) for x in re.findall(r'\[p\.(\d+)\]', ans)]
    if not cited:
        return False
    return min(abs(c - row['page']) for c in cited) <= tol


def judge(row, ans):
    """返回 (严格命中, 宽松命中, 判定标签)
    严格：答案里出现 GT 词
    宽松：严格命中，或【检索到位但没点名】—— 满足以下任一：
          · 答案引用的页码落在证据页 ±2 内（仅非 glossary 来源，page 才是正文页）
          · 答案里超出问句的部分与证据句重合 >50%
    """
    ab = is_abstain(ans)
    if row['type'] == 'unanswerable':
        return (ab, ab, 'abstain' if ab else 'FABRICATED')
    if ab:
        return (False, False, 'OVER-REFUSED')
    a = ans.lower()
    strict = any(k in a for k in row['keywords'])
    if strict:
        return (True, True, 'hit')
    loose = _page_match(row, ans) or _overlap(ans, row.get('evidence', ''), row['question']) > 0.5
    return (False, loose, 'miss-loose-hit' if loose else 'miss')


MARKER = '.built_book'


def db_owner():
    """当前 vectordb 是哪本书建的"""
    f = os.path.join(WORKDIR, MARKER)
    if os.path.exists(f):
        try:
            return io.open(f, encoding='utf-8').read().strip()
        except Exception:
            pass
    return None


def mark_db(book):
    io.open(os.path.join(WORKDIR, MARKER), 'w', encoding='utf-8').write(book)


def build_book(path, max_pages):
    """PDF 走 --pdf（会自动删库重建）；EPUB 走 --epub（append，所以先手工删库）"""
    if path.lower().endswith('.epub'):
        db = os.path.join(WORKDIR, 'vectordb')
        if os.path.isdir(db):
            shutil.rmtree(db, ignore_errors=True)
        return run([PY, MAIN, 'build', '--epub', path], BUILD_TIMEOUT)
    return run([PY, MAIN, 'build', '--pdf', path,
                '--max-pages', str(max_pages), '--no-vl'], BUILD_TIMEOUT)


def build_and_mark(path, book, max_pages):
    f = os.path.join(WORKDIR, MARKER)
    if os.path.exists(f):
        os.remove(f)              # 建库期间标记为空，中途崩了不会被误认为建好了
    out, rc = build_book(path, max_pages)
    if rc == 0:
        mark_db(book)
    return out, rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--books', required=True)
    ap.add_argument('--eval', required=True)
    ap.add_argument('--only', default='')
    ap.add_argument('--source', default='', help='只跑某来源，如 glossary')
    ap.add_argument('--limit', type=int, default=0, help='每本最多跑几题（调试）')
    ap.add_argument('--max-pages', type=int, default=999999)
    ap.add_argument('--out', default='eval_results')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--skip-build', action='store_true')
    ap.add_argument('--use-vl', action='store_true', help='开启 VL 读图（很慢，默认关）')
    ap.add_argument('--main', default='main.py', help='main.py 路径，如 E:\\Ollama_test\\code\\main.py')
    ap.add_argument('--workdir', default='', help='跑 main.py 的工作目录（向量库所在处），'
                                                 '如 E:\\Ollama_test\\data')
    ap.add_argument('--set-gate', default='', help='覆盖 ESCALATE_SIM_GATE，如 1.1762 或 none')
    ap.add_argument('--set-dynamic', default='', choices=['', 'on', 'off'],
                    help='覆盖 DYNAMIC_BUDGET：on/off')
    a = ap.parse_args()

    global MAIN, WORKDIR
    if not os.path.exists(a.main):
        print('找不到 main.py：%s\n用 --main 指定完整路径' % a.main); sys.exit(1)
    MAIN = os.path.abspath(a.main)
    WORKDIR = os.path.abspath(a.workdir) if a.workdir else (os.path.dirname(MAIN) or '.')
    if not os.path.isdir(WORKDIR):
        print('工作目录不存在：%s' % WORKDIR); sys.exit(1)
    # 消融用：把配置写进一份临时 main.py，避免手改出错
    if a.set_gate or a.set_dynamic:
        src = io.open(MAIN, encoding='utf-8').read()
        shown = []
        if a.set_gate:
            v = 'None' if a.set_gate.lower() in ('none', 'null', '') else a.set_gate
            src, n = re.subn(r'^ESCALATE_SIM_GATE\s*=.*$',
                             'ESCALATE_SIM_GATE = %s' % v, src, count=1, flags=re.M)
            if n != 1:
                print('!! 没在 main.py 里找到 ESCALATE_SIM_GATE，无法覆盖'); sys.exit(1)
            shown.append('ESCALATE_SIM_GATE=%s' % v)
        if a.set_dynamic:
            v = 'True' if a.set_dynamic == 'on' else 'False'
            src, n = re.subn(r'^DYNAMIC_BUDGET\s*=\s*\w+',
                             'DYNAMIC_BUDGET = %s' % v, src, count=1, flags=re.M)
            if n != 1:
                print('!! 没在 main.py 里找到 DYNAMIC_BUDGET，无法覆盖'); sys.exit(1)
            shown.append('DYNAMIC_BUDGET=%s' % v)
        tmp = os.path.join(WORKDIR, '_main_ablation.py')
        io.open(tmp, 'w', encoding='utf-8').write(src)
        MAIN = tmp
        print('★ 消融覆盖: %s  → 临时脚本 %s' % (' , '.join(shown), os.path.basename(tmp)))

    db = os.path.join(WORKDIR, 'vectordb')
    print('main.py  : %s' % MAIN)
    print('工作目录 : %s' % WORKDIR)
    print('向量库   : %s  %s' % (db, '(已存在，建库会覆盖)' if os.path.isdir(db) else '(将新建)'))
    if not os.path.isdir(db):
        print('  !! 这里没有 vectordb，工作目录可能指错了，指错会白跑一夜')
        if input('  确认继续？(y/N) ').strip().lower() != 'y':
            sys.exit(0)
    os.makedirs(a.out, exist_ok=True)

    BOOKS = {}
    for ext in ('*.pdf', '*.epub'):
        for p in glob.glob(os.path.join(a.books, '**', ext), recursive=True):
            BOOKS[os.path.basename(p)] = p

    summary = []
    sumpath = os.path.join(a.out, '_summary.json')
    if a.resume and os.path.exists(sumpath):
        summary = json.load(io.open(sumpath, encoding='utf-8'))

    for f in sorted(glob.glob(os.path.join(a.eval, '*.jsonl'))):
        rows = [json.loads(l) for l in io.open(f, encoding='utf-8') if l.strip()]
        if not rows:
            continue
        book = rows[0]['book']
        if a.only and a.only.lower() not in book.lower():
            continue
        if book not in BOOKS:
            print('!! 本地找不到：%s' % book)
            continue

        outp = os.path.join(a.out, re.sub(r'[^A-Za-z0-9]+', '_', book)[:60] + '.jsonl')
        if a.resume and os.path.exists(outp):
            print('跳过（已有结果）：%s' % book[:50])
            continue

        if a.source:
            rows = [r for r in rows
                    if r.get('source') == a.source or r['type'] == 'unanswerable']
        if a.limit:
            rows = rows[:a.limit]
        if not rows:
            continue

        print('\n' + '=' * 72)
        print('【%s】%d 题' % (book[:56], len(rows)), flush=True)

        if a.skip_build:
            own = db_owner()
            if own != book:
                print('  !! 跳过：--skip-build 但当前 vectordb 是【%s】建的，不是【%s】。'
                      % (own or '未知/无标记', book))
                print('     去掉 --skip-build 重建，否则数据无效。')
                continue
            print('  复用已有库（标记确认：%s）' % own)
        if not a.skip_build:
            t0 = time.time()
            print('  建库中（大部头可能要几十分钟，别关窗口）...', flush=True)
            out, rc = build_and_mark(BOOKS[book], book, a.max_pages)
            if rc != 0:
                print('  !! 建库失败，跳过。输出末尾：\n%s' % out[-400:])
                continue
            m = re.search(r'完成：[^\n]*', out)
            print('  %s' % (m.group(0) if m else '建库完成'))
            print('  建库耗时 %.1f 分钟' % ((time.time() - t0) / 60), flush=True)

        res = []
        t0 = time.time()
        for i, r in enumerate(rows, 1):
            raw, rc = run([PY, MAIN, 'ask', r['question']], ASK_TIMEOUT)
            ans, tok, esc, to = parse_ask(raw)
            ok, ok2, why = judge(r, ans)
            res.append({**r, 'answer': ans[:1500], 'ok': ok, 'ok_loose': ok2,
                        'verdict': why, 'tokens': tok, 'escalated': esc, 'timeout': to})
            if i % 20 == 0 or i == len(rows):
                print('    %d/%d  %.1f 分钟' % (i, len(rows), (time.time() - t0) / 60), flush=True)

        io.open(outp, 'w', encoding='utf-8').write(
            ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in res))

        c = Counter()
        toks = []
        for x in res:
            g = ('unans' if x['type'] == 'unanswerable'
                 else 'fuzzy' if x['type'].startswith('fuzzy') else 'ans')
            c[g + '_t'] += 1
            if x['ok']:
                c[g + '_ok'] += 1
            if x['ok_loose']:
                c[g + '_lo'] += 1
            c['v_' + x['verdict']] += 1
            if x['escalated']:
                c['esc'] += 1
            if x['tokens']:
                toks.append(x['tokens'])

        def p(g):
            return 100.0 * c[g + '_ok'] / c[g + '_t'] if c[g + '_t'] else 0.0
        med = sorted(toks)[len(toks) // 2] if toks else 0
        def q(g):
            return 100.0 * c[g + '_lo'] / c[g + '_t'] if c[g + '_t'] else 0.0
        print('  可答 严格%.1f%% 宽松%.1f%% (%d/%d) ｜ 模糊 严格%.1f%% 宽松%.1f%% (%d/%d) '
              '｜ 正确拒答 %.1f%% (%d/%d)'
              % (p('ans'), q('ans'), c['ans_ok'], c['ans_t'],
                 p('fuzzy'), q('fuzzy'), c['fuzzy_ok'], c['fuzzy_t'],
                 p('unans'), c['unans_ok'], c['unans_t']))
        print('  编造 %d ｜ 过度拒答 %d ｜ 动态升配 %d ｜ token 中位数 %d'
              % (c['v_FABRICATED'], c['v_OVER-REFUSED'], c['esc'], med), flush=True)

        summary = [s for s in summary if s['book'] != book]
        summary.append({'book': book, 'subject': rows[0]['subject'],
                        'ans_lo': c['ans_lo'], 'fuzzy_lo': c['fuzzy_lo'],
                        'ans_ok': c['ans_ok'], 'ans_t': c['ans_t'],
                        'fuzzy_ok': c['fuzzy_ok'], 'fuzzy_t': c['fuzzy_t'],
                        'unans_ok': c['unans_ok'], 'unans_t': c['unans_t'],
                        'fabricated': c['v_FABRICATED'], 'over_refused': c['v_OVER-REFUSED'],
                        'escalated': c['esc'], 'token_median': med})
        json.dump(summary, io.open(sumpath, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)

    if not summary:
        print('\n没跑成任何一本，检查路径和文件名')
        return

    print('\n' + '=' * 72)
    print('%-44s %8s %8s %8s %7s' % ('书', '可答', '模糊', '拒答', 'token'))
    t = Counter()
    for s in summary:
        for k in ('ans_ok', 'ans_t', 'fuzzy_ok', 'fuzzy_t', 'unans_ok', 'unans_t',
                  'fabricated', 'over_refused', 'escalated', 'ans_lo', 'fuzzy_lo'):
            t[k] += s[k]
        print('%-44s %7.1f%% %7.1f%% %7.1f%% %7d' % (
            s['book'][:44],
            100.0 * s['ans_ok'] / max(1, s['ans_t']),
            100.0 * s['fuzzy_ok'] / max(1, s['fuzzy_t']),
            100.0 * s['unans_ok'] / max(1, s['unans_t']),
            s['token_median']))
    print('-' * 72)
    print('%-44s %7.1f%% %7.1f%% %7.1f%%' % (
        '合计 %d 本' % len(summary),
        100.0 * t['ans_ok'] / max(1, t['ans_t']),
        100.0 * t['fuzzy_ok'] / max(1, t['fuzzy_t']),
        100.0 * t['unans_ok'] / max(1, t['unans_t'])))
    print('宽松口径：可答 %.1f%% ｜ 模糊 %.1f%%（答案复述了正确原文段落也算命中）'
          % (100.0 * t['ans_lo'] / max(1, t['ans_t']),
             100.0 * t['fuzzy_lo'] / max(1, t['fuzzy_t'])))
    print('编造 %d 道（幻觉率 %.1f%%）｜ 过度拒答 %d 道（%.1f%%）｜ 动态升配 %d 次'
          % (t['fabricated'], 100.0 * t['fabricated'] / max(1, t['unans_t']),
             t['over_refused'],
             100.0 * t['over_refused'] / max(1, t['ans_t'] + t['fuzzy_t']),
             t['escalated']))
    print('\n逐题结果在 %s\\ 下，MISS 的题带 answer 全文，可直接拉出来看' % a.out)


if __name__ == '__main__':
    main()
