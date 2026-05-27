"""
compare_finetune_failure.py — 生成 finetune 失败对比图,用于论文
对比维度:原始 best.pt vs finetune 后的权重在旧 16 类验证集上的 mAP@0.5
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 数据来源:
#   原始 = 你之前 validate.py 跑出来的结果
#   Finetune后 = 你刚才看到的 epoch1 结果
classes = [
    "ripe_apple", "unripe_apple", "rotten_apple",
    "ripe_banana", "unripe_banana", "rotten_banana",
    "ripe_grape", "unripe_grape",
    "ripe_strawberry", "unripe_strawberry",
    "ripe_persimmon", "unripe_persimmon",
    "ripe_orange", "unripe_orange", "rotten_orange",
    "rotten_fruit",
]

# 原始 best.pt 在旧验证集上的 mAP@0.5
original = [
    0.991, 0.988, 0.995,
    0.995, 0.983, 0.995,
    0.971, 0.980,
    0.995, 0.793,
    0.995, 0.994,
    0.995, 0.995, 0.995,
    0.872,
]

# Finetune 1 epoch 后的 mAP@0.5(从你截图读取)
finetuned = [
    0.716, 0.850, 0.449,
    0.732, 0.564, 0.925,
    0.595, 0.815,
    0.653, 0.646,
    0.855, 0.608,
    0.377, 0.581, 0.694,
    0.407,
]

# 计算下降幅度
drops = [(o - f) for o, f in zip(original, finetuned)]
avg_orig = sum(original) / len(original)
avg_finetuned = sum(finetuned) / len(finetuned)

# ── 绘图 ──
fig, ax = plt.subplots(figsize=(14, 6))
x = list(range(len(classes)))
width = 0.38

bars1 = ax.bar([i - width/2 for i in x], original,  width,
               label=f"原始 best.pt (avg={avg_orig:.3f})",   color="#22C55E")
bars2 = ax.bar([i + width/2 for i in x], finetuned, width,
               label=f"Finetune后 (avg={avg_finetuned:.3f})", color="#EF4444")

ax.set_xticks(x)
ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=9)
ax.set_ylabel("mAP@0.5", fontsize=11)
ax.set_title("增量微调引发灾难性遗忘 — 各类别 mAP@0.5 对比", fontsize=13, fontweight='bold')
ax.set_ylim(0, 1.05)
ax.legend(loc="lower left", fontsize=10)
ax.grid(axis="y", linestyle="--", alpha=0.5)

# 标注下降百分比
for i, (o, f) in enumerate(zip(original, finetuned)):
    drop_pct = (o - f) / o * 100
    if drop_pct > 20:
        ax.annotate(f"-{drop_pct:.0f}%", xy=(i, f), xytext=(i, f + 0.04),
                    ha='center', fontsize=8, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig("fuji_finetune/finetune_failure_comparison.png", dpi=150, bbox_inches='tight')
plt.savefig("fuji_finetune/finetune_failure_comparison.pdf", bbox_inches='tight')
print("✅ 对比图已保存:")
print("   fuji_finetune/finetune_failure_comparison.png")
print("   fuji_finetune/finetune_failure_comparison.pdf  ← 推荐放论文(矢量)")
print(f"\n📊 整体 mAP 跌幅: {avg_orig:.3f} → {avg_finetuned:.3f}  (-{(avg_orig-avg_finetuned)/avg_orig*100:.1f}%)")