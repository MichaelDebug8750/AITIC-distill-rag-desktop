# -*- coding: utf-8 -*-
"""
compare3.py —— 三路消融对照：现状 / 加闸门 / 关升配

用法：
  & "C:\\Users\\Seifer\\distill\\Scripts\\python.exe" compare3.py `
      --a eval_ab_A --b eval_ab_B --c eval_ab_C

输出：一张可直接进报告的对照表 + 结论建议
"""
import argparse, glob, io, json, os
from collections import Counter


def load(d):
    rows = []
    for f in glob.glob(os.path.join(d, '*.jsonl')):
        if '_summary' in f:
            continue
        rows += [json.loads(l) for l in io.open(f, encoding='utf-8') if l.strip()]
    return rows


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else 0


def metrics(rows):
    ans = [r for r in rows if r['type'] == 'answerable']
    fz = [r for r in rows if r['type'].startswith('fuzzy')]
    un = [r for r in rows if r['type'] == 'unanswerable']
    tk = [r['tokens'] for r in rows if r.get('tokens')]
    esc = [r for r in rows if r.get('escalated')]
    saved = [r for r in esc if r['type'] != 'unanswerable' and r['ok']]
    ovr = [r for r in rows if r['verdict'] == 'OVER-REFUSED']
    fab = [r for r in rows if r['verdict'] == 'FABRICATED']

    def p(a, b):
        return 100.0 * len(a) / len(b) if b else 0.0
    return {
        'n': len(rows),
        '可答': p([r for r in ans if r['ok']], ans),
        '模糊': p([r for r in fz if r['ok']], fz),
        '拒答': p([r for r in un if r['ok']], un),
        '幻觉': p(fab, un),
        '过度拒答': p(ovr, ans + fz),
        'token中位': med(tk),
        'token总量': sum(tk),
        '升配次数': len(esc),
        '升配救回': len(saved),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--a', required=True, help='A 现状（全升配）结果目录')
    ap.add_argument('--b', default='', help='B 加闸门结果目录')
    ap.add_argument('--c', default='', help='C 关升配结果目录')
    ap.add_argument('--out', default='消融_动态升配三路对照.md')
    a = ap.parse_args()

    cfg = [('A 现状（全升配）', a.a)]
    if a.b:
        cfg.append(('B 加闸门', a.b))
    if a.c:
        cfg.append(('C 关闭升配', a.c))

    res = []
    for name, d in cfg:
        rows = load(d)
        if not rows:
            print('!! %s 目录没数据: %s' % (name, d)); return
        m = metrics(rows)
        books = len(set(r['book'] for r in rows))
        res.append((name, books, m))

    # 一致性校验：三路必须跑的是同一批题
    ns = {m['n'] for _, _, m in res}
    if len(ns) > 1:
        print('!! 三路题数不一致 %s —— 对照无效，请确认用了同一个 eval 目录' % ns)
        return

    L = []
    def w(s=''):
        L.append(s); print(s)

    w('# 消融实验：动态升配的三路对照')
    w('')
    w('覆盖 **%d 本书 / %d 道题**，三路使用同一套题、同一份 main.py，仅切换配置。' % (res[0][1], res[0][2]['n']))
    w('')
    w('| 配置 | 可答命中 | 模糊命中 | 正确拒答 | 幻觉率 | 过度拒答 | token中位 | token总量 | 升配次数 | 升配救回 |')
    w('|---|---|---|---|---|---|---|---|---|---|')
    for name, _, m in res:
        w('| %s | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %d | %d | %d | %d |'
          % (name, m['可答'], m['模糊'], m['拒答'], m['幻觉'], m['过度拒答'],
             m['token中位'], m['token总量'], m['升配次数'], m['升配救回']))
    w('')

    base = res[0][2]
    if len(res) > 1:
        w('## 相对现状的变化')
        w('')
        w('| 配置 | 可答 | 过度拒答 | token总量 | 升配次数 |')
        w('|---|---|---|---|---|')
        for name, _, m in res[1:]:
            dt = 100.0 * (m['token总量'] - base['token总量']) / max(1, base['token总量'])
            w('| %s | %+.1fpp | %+.1fpp | %+.1f%% | %+d |'
              % (name, m['可答'] - base['可答'], m['过度拒答'] - base['过度拒答'],
                 dt, m['升配次数'] - base['升配次数']))
        w('')
        w('## 结论建议')
        w('')
        for name, _, m in res[1:]:
            dt = 100.0 * (m['token总量'] - base['token总量']) / max(1, base['token总量'])
            dov = m['过度拒答'] - base['过度拒答']
            dans = m['可答'] - base['可答']
            if dans < -2:
                v = '**不建议采用** —— 可答命中掉了 %.1fpp，代价太大' % (-dans)
            elif dt < -10 and dov < 5:
                v = '**建议采用** —— token 省 %.0f%%，过度拒答只涨 %.1fpp' % (-dt, dov)
            elif dt < -10:
                v = '**权衡后决定** —— token 省 %.0f%%，但过度拒答涨 %.1fpp' % (-dt, dov)
            else:
                v = '**收益不足** —— token 只省 %.0f%%，不值得改造' % max(0, -dt)
            w('- **%s**：%s' % (name, v))
        w('')
        w('> 判断标准：可答命中率下降超过 2pp 即视为不可接受；'
          'token 节省不足 10% 视为改造收益不足。')

    io.open(a.out, 'w', encoding='utf-8').write('\n'.join(L))
    print('\n已写入 %s' % a.out)


if __name__ == '__main__':
    main()
