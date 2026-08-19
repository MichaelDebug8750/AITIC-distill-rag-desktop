# -*- coding: utf-8 -*-
"""标定“向量侧最优距离”能否安全约束混合检索。

旧版只看基线→混合的几个指定迁移格，仍可能漏掉反向回补。本版模拟真实策略：
距离超过 T 时不用关键词结果、退回同题纯向量臂；否则采用混合臂。所有样本使用
修复了跨书复用题映射的 ``dist_full_v2.jsonl``。
"""
import os

from eval_compare import load_rows

HERE = os.path.dirname(os.path.abspath(__file__))
base = load_rows(os.path.join(HERE, "reg_rows.jsonl"))
hybrid = load_rows(os.path.join(HERE, "hyb_rows.jsonl"))
dist_rows = load_rows(os.path.join(HERE, "dist_full_v2.jsonl"))
common = sorted(set(base) & set(hybrid) & set(dist_rows))
distance = {key: float(dist_rows[key]["best"]) for key in common}

print("可用配对 %d 题（基线/混合/正确距离三者齐全）" % len(common))
print("混合臂：命中 %d，编造 %d，未命中 %d，过度拒答 %d\n" % tuple(
    sum(1 for key in common if hybrid[key].get("outcome") == outcome)
    for outcome in ("命中", "编造", "未命中", "过度拒答")))

print("=== 若规定『向量侧最优距离 > T 就不许关键词救回』===")
print("%-8s %-14s %-14s %-12s %-10s" %
      ("阈值T", "相对纯向量命中", "相对纯向量编造", "退回题数", "净值"))
thresholds = (0.85, 0.90, 0.93, 0.95, 0.99, 1.05)


def replay(keys, threshold):
    """超阈值时退回同题纯向量臂，否则采用混合臂。"""
    base_hits = sum(1 for key in keys if base[key].get("outcome") == "命中")
    base_fabrications = sum(1 for key in keys if base[key].get("outcome") == "编造")
    effective = {
        key: (base[key] if distance[key] > threshold else hybrid[key]).get("outcome")
        for key in keys
    }
    hit_delta = sum(1 for value in effective.values() if value == "命中") - base_hits
    fabrication_delta = (
        sum(1 for value in effective.values() if value == "编造") - base_fabrications)
    return hit_delta, fabrication_delta, hit_delta - 2 * fabrication_delta


best = None
for threshold in thresholds:
    hit_delta, fabrication_delta, net = replay(common, threshold)
    fallback = sum(1 for key in common if distance[key] > threshold)
    if best is None or net > best[1]:
        best = (threshold, net)
    print("%-8.2f %+14d %+14d %-12d %+10d" %
          (threshold, hit_delta, fabrication_delta, fallback, net))

print("\n最高观测净值：T=%.2f, %+d。这里不自动判定采纳；必须与同条件空白臂比较。" % best)

print("\n=== 按教材留一法，检查阈值是否只是在同一批数据上过拟合 ===")
books = sorted({key[0] for key in common})
cv_hit = cv_fabrication = cv_net = 0
chosen = []
for held_out in books:
    train = [key for key in common if key[0] != held_out]
    test = [key for key in common if key[0] == held_out]
    # 净值并列时取更小阈值：关键词救援更保守，安全侧优先。
    threshold = max(thresholds, key=lambda value: (replay(train, value)[2], -value))
    hit_delta, fabrication_delta, net = replay(test, threshold)
    chosen.append(threshold)
    cv_hit += hit_delta
    cv_fabrication += fabrication_delta
    cv_net += net
    print("  %-30s T=%.2f  命中 %+3d  编造 %+3d  净值 %+3d" %
          (held_out[:30], threshold, hit_delta, fabrication_delta, net))
print("留一法合计：命中 %+d，编造 %+d，净值 %+d；阈值选择 %s" %
      (cv_hit, cv_fabrication, cv_net, sorted(set(chosen))))
