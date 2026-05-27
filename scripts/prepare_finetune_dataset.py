"""
prepare_finetune_dataset.py — 准备 finetune 训练目录结构

策略:
  - 训练集:全部 57 张富士苹果(放到 images/train/)
  - 验证集:复用旧 fruit_dataset_v2/val/(不动它,YAML 里直接指过去)
  - 输出目录:fuji_finetune/dataset/

⚠️  此脚本只做文件复制,不修改原始数据
"""
from pathlib import Path
import shutil

SRC_IMG = Path("fuji_finetune/images")
SRC_LBL = Path("fuji_finetune/labels")

DST_ROOT = Path("fuji_finetune/dataset")
DST_IMG_TRAIN = DST_ROOT / "images" / "train"
DST_LBL_TRAIN = DST_ROOT / "labels" / "train"

DST_IMG_TRAIN.mkdir(parents=True, exist_ok=True)
DST_LBL_TRAIN.mkdir(parents=True, exist_ok=True)

imgs = sorted(SRC_IMG.glob("*.jpg"))
print(f"📦 准备复制 {len(imgs)} 张图到训练目录")

copied = 0
no_label = 0
for img_p in imgs:
    lbl_p = SRC_LBL / (img_p.stem + ".txt")
    if not lbl_p.exists() or lbl_p.read_text(encoding="utf-8").strip() == "":
        print(f"  ⚠️  跳过 {img_p.name}(无标签或空标签)")
        no_label += 1
        continue
    shutil.copy2(img_p, DST_IMG_TRAIN / img_p.name)
    shutil.copy2(lbl_p, DST_LBL_TRAIN / lbl_p.name)
    copied += 1

print(f"\n✅ 完成")
print(f"   训练集: {copied} 张  →  {DST_IMG_TRAIN}")
print(f"   跳过(无标签): {no_label} 张")
print(f"   验证集将复用 fruit_dataset_v2/val/(不动)")