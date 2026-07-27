# -*- coding: utf-8 -*-
"""
verify_eval.py —— 题库自检工具（在你本机对着原始 PDF/EPUB 重新验一遍）

不用信我的结论，这个脚本会重新打开你的书，逐题核对：
  1. 可答题   —— 目标词和每个关键词，是不是真在书里逐字出现
  2. 不可答题 —— 那个词是不是真的在全书零出现（含单复数/连字符变体）
  3. 模糊题   —— GT 词在书里，且查询句里不出现该词
  4. 共现距离 —— 关键词和目标词能不能落进同一个检索块（决定 Hit@5 真假）
  5. 结构     —— 重复题、字段缺失、编码异常、三部分是否互相泄题

用法（PowerShell，在项目根目录）：

  完整验（慢，要读全部 PDF）：
    & "C:\\Users\\Seifer\\distill\\Scripts\\python.exe" verify_eval.py `
        --books E:\\Ollama_test\\books --eval E:\\Ollama_test\\eval_by_book

  只验某一本（快，建议先这么试）：
    & "C:\\Users\\Seifer\\distill\\Scripts\\python.exe" verify_eval.py `
        --books E:\\Ollama_test\\books --eval E:\\Ollama_test\\eval_by_book `
        --only Microbiology

依赖：PyMuPDF、ebooklib、beautifulsoup4（EPUB 才需要后两个）
    pip install PyMuPDF ebooklib beautifulsoup4
"""
import argparse, glob, io, json, os, re, sys
from collections import Counter, defaultdict

WIN = 1500          # 检索块窗口，关键词与目标词超过这个距离视为不可能共现


def book_text(path):
    """把一本书读成小写全文"""
    if path.lower().endswith('.pdf'):
        import fitz
        d = fitz.open(path)
        t = ' '.join(d[i].get_text() for i in range(d.page_count))
        d.close()
        return t.lower()
    if path.lower().endswith('.epub'):
        import warnings
        warnings.filterwarnings('ignore')
        from ebooklib import epub, ITEM_DOCUMENT
        from bs4 import BeautifulSoup
        b = epub.read_epub(path)
        buf = []
        for it in b.get_items():
            if it.get_type() == ITEM_DOCUMENT:
                buf.append(BeautifulSoup(it.get_content(), 'html.parser').get_text(' ', strip=True))
        return ' '.join(buf).lower()
    return ''


def index_books(root):
    m = {}
    for ext in ('*.pdf', '*.epub'):
        for p in glob.glob(os.path.join(root, '**', ext), recursive=True):
            m[os.path.basename(p)] = p
    return m


def variants(t):
    return {t, t + 's', t + 'es', t.rstrip('s'), t.replace(' ', '-'), t.replace('-', ' ')}


def cooccur(low, a, b, win=WIN):
    pa = [m.start() for m in re.finditer(re.escape(a), low)][:400]
    pb = [m.start() for m in re.finditer(re.escape(b), low)][:400]
    if not pa or not pb:
        return False
    for x in pa:
        for y in pb:
            if abs(x - y) < win:
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--books', required=True, help='书籍根目录')
    ap.add_argument('--eval', required=True, help='eval_by_book 目录')
    ap.add_argument('--only', default='', help='只验文件名含该字符串的书')
    ap.add_argument('--skip-cooccur', action='store_true', help='跳过共现检查（更快）')
    a = ap.parse_args()

    BOOKS = index_books(a.books)
    files = sorted(glob.glob(os.path.join(a.eval, '*.jsonl')))
    if not files:
        print('没找到题集文件，检查 --eval 路径'); sys.exit(1)

    R = Counter()
    problems = []
    missing_books = []

    for f in files:
        rows = [json.loads(l) for l in io.open(f, encoding='utf-8') if l.strip()]
        if not rows:
            continue
        book = rows[0]['book']
        if a.only and a.only.lower() not in book.lower():
            continue
        if book not in BOOKS:
            missing_books.append(book)
            continue
        print('读取 %s ...' % book[:60], flush=True)
        low = book_text(BOOKS[book])
        if not low:
            problems.append(('读不出文本', book, '')); continue

        seen_q, ans_terms, fz_terms = set(), set(), set()
        for r in rows:
            t = r['type']
            q = r['question'].strip().lower()
            if q in seen_q:
                problems.append(('同书重复题', book, r['question'][:60]))
            seen_q.add(q)
            if r.get('expect') not in ('answer', 'abstain'):
                problems.append(('expect 非法', book, str(r.get('expect'))))
            if re.search(r'[\ufffd\u4e00-\u9fff]', r['question']):
                problems.append(('题干字符异常', book, r['question'][:50]))

            if t == 'answerable':
                R['可答_总'] += 1
                ans_terms.add(r['keywords'][0] if r['keywords'] else '')
                if not r['keywords']:
                    problems.append(('可答题无关键词', book, r['question'][:50])); continue
                miss = [k for k in r['keywords'] if k not in low]
                if r['term'].lower() not in low:
                    problems.append(('目标词不在书里', book, r['term']))
                elif miss:
                    problems.append(('关键词不在书里', book, '%s -> %s' % (r['term'], miss)))
                else:
                    R['可答_通过'] += 1
                    if not a.skip_cooccur:
                        for k in r['keywords'][1:]:
                            if not cooccur(low, r['keywords'][0], k):
                                problems.append(('关键词与目标词距离过远', book,
                                                 '%s / %s' % (r['keywords'][0], k)))
                                R['共现失败'] += 1

            elif t == 'unanswerable':
                R['不可答_总'] += 1
                if r.get('keywords'):
                    problems.append(('不可答题不该有关键词', book, r['question'][:50]))
                tt = (r.get('term') or '').lower()
                leak = [v for v in variants(tt) if len(v) > 4 and v in low] if tt else []
                if leak:
                    problems.append(('探针泄漏（书里有这个词）', book, '%s -> %s' % (tt, leak[:2])))
                else:
                    R['不可答_通过'] += 1

            else:
                R['模糊_总'] += 1
                fz_terms.add(r['keywords'][0] if r['keywords'] else '')
                if not r['keywords']:
                    problems.append(('模糊题无 GT', book, r['question'][:50])); continue
                gt = r['keywords'][0]
                if gt not in low:
                    problems.append(('模糊题 GT 不在书里', book, gt))
                elif gt in r['question'].lower():
                    problems.append(('模糊题查询里泄漏了 GT', book, r['question'][:60]))
                else:
                    R['模糊_通过'] += 1

        overlap = (ans_terms & fz_terms) - {''}
        if overlap:
            problems.append(('可答题与模糊题目标词重叠', book, str(list(overlap)[:3])))

    print('\n' + '=' * 68)
    print('可答题   %d / %d' % (R['可答_通过'], R['可答_总']))
    print('不可答题 %d / %d' % (R['不可答_通过'], R['不可答_总']))
    print('模糊题   %d / %d' % (R['模糊_通过'], R['模糊_总']))
    if missing_books:
        print('\n本地找不到对应书籍文件 %d 本：' % len(missing_books))
        for b in missing_books[:10]:
            print('   ', b)
    if problems:
        c = Counter(p[0] for p in problems)
        print('\n发现问题 %d 处：' % len(problems))
        for k, v in c.most_common():
            print('   %-28s %d' % (k, v))
        print('\n前 20 条明细：')
        for p in problems[:20]:
            print('   [%s] %s | %s' % (p[0], p[1][:40], p[2][:60]))
        with io.open('verify_eval_报告.txt', 'w', encoding='utf-8') as fp:
            for p in problems:
                fp.write('%s\t%s\t%s\n' % p)
        print('\n完整明细已写入 verify_eval_报告.txt')
    else:
        print('\n结构与内容检查：无问题')

    ok = (R['可答_通过'] == R['可答_总'] and R['不可答_通过'] == R['不可答_总']
          and R['模糊_通过'] == R['模糊_总'] and not problems)
    print('\n最终判定：' + ('PASS' if ok else 'FAIL'))
    sys.exit(0 if ok else 2)


if __name__ == '__main__':
    main()
