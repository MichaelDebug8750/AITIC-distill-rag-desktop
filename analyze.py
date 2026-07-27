# -*- coding: utf-8 -*-
"""
analyze.py —— 把评测结果拆开看：MISS 是检索问题还是判定问题

用法：
  & "C:\\Users\\Seifer\\distill\\Scripts\\python.exe" analyze.py eval_results\\Microbiology_pdf.jsonl
  # 只看某一类
  ... analyze.py eval_results\\Microbiology_pdf.jsonl --type fuzzy_desc
"""
import argparse, io, json, re, sys
from collections import Counter


def short(s, n=170):
    s = re.sub(r'\s+', ' ', (s or '')).strip()
    return s[:n] + ('…' if len(s) > n else '')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('path')
    ap.add_argument('--type', default='')
    ap.add_argument('--max', type=int, default=100)
    a = ap.parse_args()

    rows = [json.loads(l) for l in io.open(a.path, encoding='utf-8') if l.strip()]
    if a.type:
        rows = [r for r in rows if r['type'] == a.type]

    c = Counter(r['verdict'] for r in rows)
    print('共 %d 题：%s\n' % (len(rows), dict(c)))

    # ===== 1. 编造（最该看的）=====
    fab = [r for r in rows if r['verdict'] == 'FABRICATED']
    if fab:
        print('=' * 70)
        print('【编造 %d 道】书里没有却答了 —— 幻觉，最该改的' % len(fab))
        for r in fab[:a.max]:
            print('\n  探针词: %s  (%s)' % (r['term'], r.get('probe_class', '-')))
            print('  问: %s' % short(r['question'], 100))
            print('  答: %s' % short(r['answer']))

    # ===== 2. 过度拒答 =====
    over = [r for r in rows if r['verdict'] == 'OVER-REFUSED']
    if over:
        print('\n' + '=' * 70)
        print('【过度拒答 %d 道】书里有却说没有' % len(over))
        for r in over[:a.max]:
            print('\n  [%s] %s' % (r['type'], short(r['question'], 100)))
            print('  GT=%s  在书里出现 %s 次' % (r['keywords'][0], r.get('term_freq', '?')))
            print('  证据: %s' % short(r.get('evidence', ''), 110))

    # ===== 3. MISS：分「答非所问」和「答对了但没说出那个词」=====
    miss = [r for r in rows if r['verdict'] == 'miss']
    if miss:
        print('\n' + '=' * 70)
        print('【MISS %d 道】答了但没命中 GT —— 要分两种看' % len(miss))
        near = far = 0
        for r in miss[:a.max]:
            ans = (r['answer'] or '').lower()
            ev = (r.get('evidence') or '').lower()
            # 证据句里的实词有多少出现在答案里 → 高说明检索对了，只是没说出 GT 那个词
            ws = [w for w in re.findall(r'[a-z]{5,}', ev) if w not in ans]
            evw = [w for w in re.findall(r'[a-z]{5,}', ev)]
            overlap = 1 - (len(ws) / len(evw)) if evw else 0
            tag = '判定过严？（检索像是对的）' if overlap > 0.35 else '真没检索到'
            if overlap > 0.35:
                near += 1
            else:
                far += 1
            print('\n  [%s | %s] 内容重合 %.0f%%' % (r['type'], tag, overlap * 100))
            print('  问: %s' % short(r['question'], 110))
            print('  GT=%s (书中 %s 次)' % (r['keywords'][0], r.get('term_freq', '?')))
            print('  答: %s' % short(r['answer'], 150))
        print('\n  小结：判定过严 %d 道 ｜ 真没检索到 %d 道' % (near, far))

    # ===== 4. 按题型汇总 =====
    print('\n' + '=' * 70)
    g = {}
    for r in rows:
        k = r['type']
        g.setdefault(k, [0, 0])
        g[k][1] += 1
        if r['ok']:
            g[k][0] += 1
    for k, (ok, t) in sorted(g.items()):
        print('  %-14s %d/%d  %.1f%%' % (k, ok, t, 100.0 * ok / t))
    esc = sum(1 for r in rows if r.get('escalated'))
    toks = sorted(r['tokens'] for r in rows if r.get('tokens'))
    print('  动态升配 %d/%d (%.0f%%) ｜ token 中位 %d / 最大 %d'
          % (esc, len(rows), 100.0 * esc / len(rows),
             toks[len(toks) // 2] if toks else 0, toks[-1] if toks else 0))


if __name__ == '__main__':
    main()
