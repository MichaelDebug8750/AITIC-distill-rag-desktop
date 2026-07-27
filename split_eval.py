# -*- coding: utf-8 -*-
"""
split_eval.py —— 从 eval_ALL.jsonl 拆出 eval_by_book\ 单本题集

verify_eval.py 和 run_eval_batch.py 都要按「一个文件一本书」来跑，
你下的 zip 里只有学科合并版，用这个脚本本地拆一下就有了。

用法（在 E:\\Ollama_test 下）：
  & "C:\\Users\\Seifer\\distill\\Scripts\\python.exe" split_eval.py `
      --all E:\\Ollama_test\\eval\\eval_ALL.jsonl --out E:\\Ollama_test\\eval\\eval_by_book
"""
import argparse, io, json, os, re
from collections import defaultdict, Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--all', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    if not os.path.exists(a.all):
        print('找不到 %s' % a.all); return
    os.makedirs(a.out, exist_ok=True)

    by = defaultdict(list)
    bad = 0
    for i, l in enumerate(io.open(a.all, encoding='utf-8'), 1):
        if not l.strip():
            continue
        try:
            r = json.loads(l)
        except Exception as e:
            bad += 1
            print('  第 %d 行解析失败: %s' % (i, e))
            continue
        by[(r['subject'], r['book'])].append(r)

    for (subj, book), rows in sorted(by.items()):
        name = '%s__%s.jsonl' % (subj.replace(' ', ''),
                                 re.sub(r'[^A-Za-z0-9]+', '_',
                                        os.path.splitext(book)[0])[:46].strip('_'))
        io.open(os.path.join(a.out, name), 'w', encoding='utf-8').write(
            ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in rows))

    n = sum(len(v) for v in by.values())
    print('拆出 %d 本 / %d 道题 -> %s' % (len(by), n, a.out))
    if bad:
        print('!! 有 %d 行坏行，检查 eval_ALL.jsonl 是否下载完整' % bad)

    c = Counter()
    for rows in by.values():
        for r in rows:
            c[r['type']] += 1
    print('   可答 %d ｜ 不可答 %d ｜ 模糊 %d'
          % (c['answerable'], c['unanswerable'], c['fuzzy_desc'] + c['fuzzy_kw']))
    if len(by) != 55 or n != 4454:
        print('!! 数量对不上（应为 55 本 / 4454 题），eval_ALL.jsonl 可能没下全')
    else:
        print('   数量核对通过：55 本 / 4454 题')


if __name__ == '__main__':
    main()
