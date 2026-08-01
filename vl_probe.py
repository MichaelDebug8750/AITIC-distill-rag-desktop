# -*- coding: utf-8 -*-
"""
vl_probe.py —— 诊断 VL 块的检索召回情况

背景：图内题实测发现，VL 已解析出的图内标签（如 sphenoidal fontanelle）
      在问答中被拒答。需要判断是「VL 没入库」还是「入库了但检索不到」。

本脚本直接查询向量库，统计 top-k 里 type=figure 的块占比，不调 LLM，很快。

用法：
  & "C:\\Users\\Seifer\\distill\\Scripts\\python.exe" vl_probe.py `
      --main E:\\Ollama_test\\data\\main.py --workdir E:\\Ollama_test\\data
"""
import argparse, io, json, os, sys, importlib.util
from collections import Counter


def load_main(path):
    spec = importlib.util.spec_from_file_location('dm', path)
    m = importlib.util.module_from_spec(spec)
    sys.argv = ['probe']
    spec.loader.exec_module(m)
    return m


# 三组查询：图内独有术语 / 图内标签原文 / 普通文本问题（对照）
PROBES_FIGURE_ONLY = [
    "sphenoidal fontanelle",
    "mastoid fontanelle",
    "infrapatellar bursa",
    "Which fontanelle is located near the sphenoid bone in the fetal skull?",
    "Which bursa lies below the patella?",
    "What are the fontanelles labeled in the fetal skull diagram?",
]
PROBES_FIGURE_STYLE = [
    "Extracted Text Labels from Figure",
    "labels in the diagram of the scapula",
    "text labels shown in the knee joint figure",
    "structures labeled in the hip bone diagram",
]
PROBES_TEXT = [
    "What is the deltoid tuberosity?",
    "Define osteoarthritis.",
    "What is the function of the patellar ligament?",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--main', required=True)
    ap.add_argument('--workdir', default='')
    ap.add_argument('--topk', type=int, default=0, help='默认用 main.py 的 TOP_K')
    a = ap.parse_args()

    wd = os.path.abspath(a.workdir) if a.workdir else os.path.dirname(os.path.abspath(a.main))
    os.chdir(wd)
    M = load_main(os.path.abspath(a.main))
    import chromadb
    col = chromadb.PersistentClient(path=M.DB_PATH).get_collection(M.COLLECTION)
    K = a.topk or M.TOP_K

    total = col.count()
    print('库内块数: %d' % total)

    # ---- 1. 库里到底有多少 VL 块 ----
    got = col.get(include=['metadatas'], limit=total)
    tc = Counter(m.get('type', '?') for m in got['metadatas'])
    print('块类型分布: %s' % dict(tc))
    nfig = tc.get('figure', 0)
    print('VL(figure) 块占比: %d/%d = %.2f%%\n' % (nfig, total, 100.0 * nfig / total))
    if nfig == 0:
        print('!! 库里没有 figure 块 —— 当前是纯文本库，请先用 --vl-limit/--vl-from 建 VL 库。')
        return

    # figure 块分布在哪些页
    figpages = sorted({m.get('page') for m in got['metadatas'] if m.get('type') == 'figure'})
    print('figure 块覆盖页码: %s\n' % figpages)

    # ---- 2. 三组查询的召回情况 ----
    def probe(title, queries):
        print('=' * 66)
        print('【%s】' % title)
        hit_any = 0
        for q in queries:
            qv = M.embed([q])[0]
            res = col.query(query_embeddings=[qv], n_results=K)
            metas = res['metadatas'][0]
            dists = (res.get('distances') or [[]])[0]
            types = [m.get('type', '?') for m in metas]
            nf = sum(1 for t in types if t == 'figure')
            if nf:
                hit_any += 1
            # 最优 figure 块排在第几
            rank = next((i + 1 for i, t in enumerate(types) if t == 'figure'), None)
            print('  %-52s figure %d/%d%s' %
                  (q[:52], nf, K,
                   ('，最高排名第 %d（dist=%.4f）' % (rank, dists[rank - 1])) if rank else '，一个都没召回'))
        print('  → %d/%d 个查询召回了至少一个 figure 块\n' % (hit_any, len(queries)))
        return hit_any

    a1 = probe('图内独有术语（正文零出现，只有 VL 块里有）', PROBES_FIGURE_ONLY)
    a2 = probe('图注风格查询（贴近 VL 块的文本形态）', PROBES_FIGURE_STYLE)
    a3 = probe('普通文本问题（对照组，figure 召回少才正常）', PROBES_TEXT)

    print('=' * 66)
    print('诊断结论：')
    if a1 == 0:
        print('  ❌ 图内独有术语一个 figure 块都召回不到 —— VL 块入库了但检索不到。')
        print('     可能原因：VL 文本形态（"### Extracted Text Labels..."）与自然问句')
        print('     语义距离过大，embedding 无法对齐。')
    elif a1 < len(PROBES_FIGURE_ONLY) / 2:
        print('  ⚠️ 图内术语的 figure 召回率偏低（%d/%d）—— VL 内容进了检索空间但竞争不过文本块。'
              % (a1, len(PROBES_FIGURE_ONLY)))
    else:
        print('  ✅ 图内术语能召回 figure 块（%d/%d）—— 检索正常，问题在生成端。'
              % (a1, len(PROBES_FIGURE_ONLY)))
    if a2 > a1:
        print('  · 图注风格查询召回更好（%d vs %d）→ 印证「形态不匹配」假设：' % (a2, a1))
        print('    VL 块能被检索到，但只在查询本身像图注时才行。')
    print('  · 对照组（普通文本题）figure 召回 %d/%d，符合预期即可。'
          % (a3, len(PROBES_TEXT)))


if __name__ == '__main__':
    main()
