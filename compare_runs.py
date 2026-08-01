# -*- coding: utf-8 -*-
r"""
compare_runs.py —— 用同一段代码统计两轮评测，避免"口径不同"造成的假差异

起因：v6 排查全程都在被"数字来源不一致"干扰。项目记录里的 v3full 指标
     （可答 90.6% / 模糊 40.3% / 幻觉 2.4%）是当时的脚本算的，
     和现在的统计口径是否完全一致，没人验证过。
     这个脚本对两个目录用完全相同的逻辑重算，并逐题比对翻转。

用法：
    python compare_runs.py E:\Ollama_test\eval_results_v3full E:\Ollama_test\eval_results_v7full
    python compare_runs.py <A> <B> --flips 40        # 多列几道翻转题
"""
import json, io, os, glob, sys, argparse, statistics
from collections import Counter, defaultdict


def load(d):
    rows = []
    for f in glob.glob(os.path.join(d, '*.jsonl')):
        if os.path.basename(f).startswith('_'):
            continue
        for l in io.open(f, encoding='utf-8'):
            l = l.strip()
            if l:
                rows.append(json.loads(l))
    return rows


def metrics(rows):
    m = {}
    for t in ('answerable', 'fuzzy_kw', 'fuzzy_desc', 'unanswerable'):
        s = [r for r in rows if r['type'] == t]
        m[t] = (sum(r['ok'] for r in s), len(s), sum(r['ok_loose'] for r in s))
    fz = [r for r in rows if r['type'].startswith('fuzzy')]
    m['fuzzy_all'] = (sum(r['ok'] for r in fz), len(fz), sum(r['ok_loose'] for r in fz))
    v = Counter(r['verdict'] for r in rows)
    m['fabricated'] = v.get('FABRICATED', 0)
    m['over_refused'] = v.get('OVER-REFUSED', 0)
    m['escalated'] = sum(r['escalated'] for r in rows)
    m['n'] = len(rows)
    m['token_median'] = statistics.median([r['tokens'] for r in rows]) if rows else 0
    m['timeout'] = sum(r['timeout'] for r in rows)
    return m


def pct(a, b):
    return 100.0 * a / b if b else 0.0


def line(label, A, B, key, denom_key=None):
    a_ok, a_t, _ = A[key]
    b_ok, b_t, _ = B[key]
    if a_t != b_t:
        note = '  !! 题数不同 %d vs %d' % (a_t, b_t)
    else:
        note = ''
    pa, pb = pct(a_ok, a_t), pct(b_ok, b_t)
    print('  %-12s %5d/%-5d %6.1f%%   %5d/%-5d %6.1f%%   %+6.1fpp%s'
          % (label, a_ok, a_t, pa, b_ok, b_t, pb, pb - pa, note))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dir_a'); ap.add_argument('dir_b')
    ap.add_argument('--flips', type=int, default=20)
    a = ap.parse_args()

    RA, RB = load(a.dir_a), load(a.dir_b)
    if not RA or not RB:
        sys.exit('[错误] 有一边没读到 jsonl：A=%d B=%d' % (len(RA), len(RB)))
    A, B = metrics(RA), metrics(RB)

    print('A = %s  (%d 题)' % (a.dir_a, A['n']))
    print('B = %s  (%d 题)' % (a.dir_b, B['n']))
    for d, tag in ((a.dir_a, 'A'), (a.dir_b, 'B')):
        fp = os.path.join(d, '_fingerprint.json')
        if os.path.exists(fp):
            j = json.load(io.open(fp, encoding='utf-8')).get('runtime', {})
            print('  %s 自证: prompt=%s gate=%s ollama=%s' % (
                tag, j.get('prompt_variant'), j.get('escalate_sim_gate'),
                (j.get('env') or {}).get('ollama_server')))
        else:
            print('  %s 自证: 无 _fingerprint.json（该轮跑于自证机制之前）' % tag)

    print('\n%-14s %20s %20s %10s' % ('', 'A', 'B', '差'))
    line('可答', A, B, 'answerable')
    line('模糊 kw', A, B, 'fuzzy_kw')
    line('模糊 desc', A, B, 'fuzzy_desc')
    line('模糊 合计', A, B, 'fuzzy_all')
    line('不可答', A, B, 'unanswerable')
    un_a, un_b = A['unanswerable'][1], B['unanswerable'][1]
    print('  %-12s %5d       %6.1f%%   %5d       %6.1f%%   %+6.1fpp'
          % ('幻觉', A['fabricated'], pct(A['fabricated'], un_a),
             B['fabricated'], pct(B['fabricated'], un_b),
             pct(B['fabricated'], un_b) - pct(A['fabricated'], un_a)))
    print('  %-12s %5d       %6.1f%%   %5d       %6.1f%%'
          % ('过度拒答', A['over_refused'], pct(A['over_refused'], A['n']),
             B['over_refused'], pct(B['over_refused'], B['n'])))
    print('  %-12s %5d       %6.1f%%   %5d       %6.1f%%'
          % ('升配', A['escalated'], pct(A['escalated'], A['n']),
             B['escalated'], pct(B['escalated'], B['n'])))
    print('  %-12s %5.0f              %5.0f' % ('token中位', A['token_median'], B['token_median']))
    if A['timeout'] or B['timeout']:
        print('  !! 超时 A=%d B=%d' % (A['timeout'], B['timeout']))

    # ---- 逐题翻转 ----
    ka = {(r['book'], r['question']): r for r in RA}
    kb = {(r['book'], r['question']): r for r in RB}
    common = set(ka) & set(kb)
    print('\n同题 %d 道（A独有 %d，B独有 %d）' % (len(common), len(ka) - len(common), len(kb) - len(common)))
    if len(common) < min(len(ka), len(kb)) * 0.95:
        print('  !! 同题率偏低，两轮题集可能不同，逐题比对仅供参考')

    win, lose = [], []
    for k in common:
        x, y = ka[k], kb[k]
        if not x['ok'] and y['ok']:
            win.append((k, x, y))
        elif x['ok'] and not y['ok']:
            lose.append((k, x, y))
    print('B 相对 A：救回 %d 道，改坏 %d 道，净 %+d' % (len(win), len(lose), len(win) - len(lose)))
    for tag, pool in (('救回', win), ('改坏', lose)):
        print('\n  -- %s 按题型 --' % tag)
        for t, c in Counter(x['type'] for _, x, _ in pool).most_common():
            print('     %-14s %d' % (t, c))

    # ---- 改坏归因（自动分类，避免靠眼睛抽样下结论）----
    if lose:
        import re as _re
        AB = _re.compile(r"NO\s+REFERENCE\s+FOUND", _re.I)
        def strip_tag(t): return _re.sub(r"\[[^\]]*\]", "", t or "").strip()
        cats = defaultdict(list)
        for k, x, y in lose:
            ax, by = x['answer'] or '', y['answer'] or ''
            a_ref, b_ref = bool(AB.search(ax)), bool(AB.search(by))
            la, lb = len(strip_tag(ax)), len(strip_tag(by))
            if a_ref and not b_ref:
                c = '① A拒答 -> B作答（幻觉风险）'
            elif b_ref and not a_ref:
                c = '② A作答 -> B拒答（漏答）'
            elif la and la < 110 and lb > la * 1.6:
                c = '③ A短答报术语 -> B长句复述（关键词丢失）'
            else:
                c = '④ 两边都作答，措辞变化'
            cats[c].append((k, x, y))
        print('\n  == 改坏归因（共 %d 道）==' % len(lose))
        for c in sorted(cats):
            pool = cats[c]
            bytype = Counter(x['type'] for _, x, _ in pool)
            print('     %-34s %3d 道   %s' % (c, len(pool),
                  ' '.join('%s=%d' % (t, n) for t, n in bytype.most_common())))
        # ③ 单列，这是 V3 指令的直接副作用，最值得看
        key3 = '③ A短答报术语 -> B长句复述（关键词丢失）'
        if cats.get(key3):
            print('\n     -- ③ 明细（最多 12 道）--')
            for (bk, q), x, y in cats[key3][:12]:
                print('        %-24s | kw=%s' % (bk[:24], (x.get('keywords') or [])[:3]))
                print('           A(%3d): %s' % (len(strip_tag(x['answer'])), x['answer'][:88].replace('\n', ' ')))
                print('           B(%3d): %s' % (len(strip_tag(y['answer'])), y['answer'][:88].replace('\n', ' ')))

        print('\n  -- 改坏的题（最多列 %d 道）--' % a.flips)
        for (bk, q), x, y in lose[:a.flips]:
            print('     %-26s | %s' % (bk[:26], q[:56]))
            print('        A: %s' % x['answer'][:100].replace('\n', ' '))
            print('        B: %s' % y['answer'][:100].replace('\n', ' '))


if __name__ == '__main__':
    main()
