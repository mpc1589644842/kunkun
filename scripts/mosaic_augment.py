"""
mosaic_augment.py
把现有单果图4张拼成mosaic多果图，自动生成对应标注
运行：python mosaic_augment.py
"""

import cv2
import random
import shutil
from pathlib import Path

DATASET_ROOT  = "./fruit_dataset"
OUTPUT_SIZE   = 640
NUM_MOSAIC    = 600   # 生成600张，占训练集约20%
SEED          = 42

CLASS_NAMES = [
    "ripe_apple", "rotten_apple",
    "ripe_banana", "rotten_banana", "unripe_banana",
    "ripe_orange", "rotten_orange", "unripe_orange",
]

random.seed(SEED)

def load_samples():
    """加载所有训练图及其标注，兼容子目录和平铺两种结构"""
    img_root = Path(DATASET_ROOT) / "images" / "train"
    lbl_root = Path(DATASET_ROOT) / "labels" / "train"
    samples  = []

    # 平铺结构：images/train/*.jpg
    for img in img_root.glob("*.[jp][pn]g"):
        lbl = lbl_root / (img.stem + ".txt")
        if lbl.exists():
            samples.append((img, lbl))

    # 子目录结构：images/train/{cls_name}/*.jpg
    for cls_name in CLASS_NAMES:
        cls_img_dir = img_root / cls_name
        cls_lbl_dir = lbl_root / cls_name
        if not cls_img_dir.exists():
            continue
        for img in cls_img_dir.glob("*.[jp][pn]g"):
            # 优先找平铺标签（relabel.py生成的带前缀文件名）
            lbl = lbl_root / f"{cls_name}_{img.stem}.txt"
            if not lbl.exists():
                lbl = cls_lbl_dir / (img.stem + ".txt")
            if lbl.exists():
                samples.append((img, lbl))

    print(f"  加载训练样本：{len(samples)} 张")
    return samples

def read_labels(lbl_path):
    boxes = []
    for line in Path(lbl_path).read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            boxes.append([int(parts[0])] + [float(x) for x in parts[1:]])
    return boxes

def make_mosaic(samples):
    half   = OUTPUT_SIZE // 2
    canvas = __import__("numpy").zeros((OUTPUT_SIZE, OUTPUT_SIZE, 3), dtype=__import__("numpy").uint8)
    chosen = random.sample(samples, 4)
    new_boxes = []

    offsets = [(0, 0), (half, 0), (0, half), (half, half)]

    for i, (img_path, lbl_path) in enumerate(chosen):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, (half, half))
        ox, oy = offsets[i]
        canvas[oy:oy+half, ox:ox+half] = img

        for box in read_labels(lbl_path):
            cls_id, cx, cy, w, h = box
            new_cx = (ox + cx * half) / OUTPUT_SIZE
            new_cy = (oy + cy * half) / OUTPUT_SIZE
            new_w  = w * half / OUTPUT_SIZE
            new_h  = h * half / OUTPUT_SIZE
            x1 = max(0.0, new_cx - new_w / 2)
            y1 = max(0.0, new_cy - new_h / 2)
            x2 = min(1.0, new_cx + new_w / 2)
            y2 = min(1.0, new_cy + new_h / 2)
            if x2 - x1 < 0.01 or y2 - y1 < 0.01:
                continue
            new_boxes.append([
                cls_id,
                (x1 + x2) / 2, (y1 + y2) / 2,
                x2 - x1,       y2 - y1
            ])

    return canvas, new_boxes

if __name__ == "__main__":
    import numpy as np
    samples  = load_samples()
    if len(samples) < 4:
        print("训练样本不足4张，退出")
        exit()

    out_img = Path(DATASET_ROOT) / "images" / "train"
    out_lbl = Path(DATASET_ROOT) / "labels" / "train"

    success = 0
    for i in range(NUM_MOSAIC):
        canvas, boxes = make_mosaic(samples)
        if not boxes:
            continue
        name = f"mosaic_{i:04d}"
        cv2.imwrite(str(out_img / f"{name}.jpg"), canvas)
        Path(out_lbl / f"{name}.txt").write_text(
            "\n".join(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}" for b in boxes)
        )
        success += 1
        if (i + 1) % 100 == 0:
            print(f"  已生成 {i+1}/{NUM_MOSAIC}")

    print(f"\n✅ mosaic合成完成，共生成 {success} 张")
    print(f"   训练集多目标图占比约：{success}/{len(samples)+success} = {success/(len(samples)+success)*100:.1f}%")