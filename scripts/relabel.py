"""
relabel.py
功能：
  1. 单果居中图 → 写入保守居中框 0.5 0.5 0.75 0.75
  2. 香蕉多果图（文件名含 bunch/multi 或框数据疑似多目标）→ 用预训练模型伪标注
  3. 标签目录结构拍平为 labels/train/*.txt（去掉按类别分子文件夹）
     使 images/train/ 和 labels/train/ 一一对应

运行前提：pip install ultralytics
运行：python tools/relabel.py
"""

import shutil
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ── 配置 ──────────────────────────────────────
DATASET_ROOT  = "./fruit_dataset"
SPLITS        = ["train", "val", "test"]
# 单果居中框：宽高各占75%，比1.0更接近真实目标边界
SINGLE_BOX    = "0.5 0.5 0.75 0.75"
# 香蕉多果图判断：文件名包含以下关键词时触发伪标注
MULTI_KEYWORDS = ["bunch", "multi", "cluster", "多"]
# 伪标注用预训练模型（只用来做香蕉多果图的框定位，不影响最终训练类别）
PRETRAIN_MODEL = "yolo11n.pt"

CLASS_MAP = {
    "ripe_apple": 0, "rotten_apple": 1,
    "ripe_banana": 2, "rotten_banana": 3, "unripe_banana": 4,
    "ripe_orange": 5, "rotten_orange": 6, "unripe_orange": 7,
}
# COCO类别中香蕉的class_id=46，苹果=47，橙子=49（用于伪标注过滤）
COCO_FRUIT_IDS = {46, 47, 49}

# ── 工具函数 ──────────────────────────────────
def is_multi_fruit_img(img_path: Path) -> bool:
    """判断是否为多果图（根据文件名关键词）"""
    name_lower = img_path.stem.lower()
    return any(kw in name_lower for kw in MULTI_KEYWORDS)

def pseudo_label_multi(img_path: Path, cls_id: int, model) -> list[str]:
    """
    用预训练COCO模型对多果图做伪标注，返回YOLO格式行列表
    若检测失败则fallback到单果框
    """
    results = model.predict(str(img_path), conf=0.25, verbose=False)
    boxes   = results[0].boxes
    lines   = []
    if boxes is not None:
        for box in boxes:
            coco_cls = int(box.cls.item())
            if coco_cls not in COCO_FRUIT_IDS:
                continue
            cx, cy, w, h = box.xywhn[0].tolist()
            # 过滤过小的框（噪声）
            if w < 0.05 or h < 0.05:
                continue
            lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    # fallback
    if not lines:
        lines = [f"{cls_id} {SINGLE_BOX}"]
    return lines

def get_flat_img_list(split: str) -> list[tuple[Path, int]]:
    """
    遍历 images/{split}/{cls_name}/*.jpg 结构
    返回 [(img_path, cls_id), ...]
    """
    img_root = Path(DATASET_ROOT) / "images" / split
    result   = []
    for cls_name, cls_id in CLASS_MAP.items():
        cls_dir = img_root / cls_name
        if not cls_dir.exists():
            continue
        for img in sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png")):
            result.append((img, cls_id))
    return result

# ── 主流程 ────────────────────────────────────
def relabel(split: str, pseudo_model=None):
    samples  = get_flat_img_list(split)
    lbl_root = Path(DATASET_ROOT) / "labels" / split
    lbl_root.mkdir(parents=True, exist_ok=True)

    single_count = 0
    pseudo_count = 0
    error_count  = 0

    for img_path, cls_id in samples:
        # 输出标签路径：拍平到 labels/{split}/{img_stem}.txt
        # 注意：若不同类别下有同名文件，加类别前缀避免覆盖
        cls_name = [k for k, v in CLASS_MAP.items() if v == cls_id][0]
        txt_name = f"{cls_name}_{img_path.stem}.txt"
        txt_path = lbl_root / txt_name

        try:
            if is_multi_fruit_img(img_path) and pseudo_model is not None:
                lines = pseudo_label_multi(img_path, cls_id, pseudo_model)
                pseudo_count += 1
            else:
                lines = [f"{cls_id} {SINGLE_BOX}"]
                single_count += 1

            txt_path.write_text("\n".join(lines))

        except Exception as e:
            print(f"  ⚠️  {img_path.name}: {e}")
            error_count += 1

    print(f"[{split}] 完成：单果框={single_count} | 伪标注={pseudo_count} | 错误={error_count}")
    return single_count + pseudo_count

def flatten_images(split: str):
    """
    把 images/{split}/{cls_name}/xxx.jpg 复制到 images/{split}/xxx.jpg
    同时重命名为 {cls_name}_{stem}.jpg 避免冲突
    原子目录保留不删除
    """
    img_root = Path(DATASET_ROOT) / "images" / split
    flat_dir = img_root  # 直接写到同级，靠前缀区分

    moved = 0
    for cls_name, cls_id in CLASS_MAP.items():
        cls_dir = img_root / cls_name
        if not cls_dir.exists():
            continue
        for img in sorted(cls_dir.glob("*.jpg")) + sorted(cls_dir.glob("*.png")):
            dst = flat_dir / f"{cls_name}_{img.name}"
            if not dst.exists():
                shutil.copy2(img, dst)
                moved += 1
    print(f"[{split}] 图片拍平完成，新增 {moved} 个文件到 images/{split}/")

if __name__ == "__main__":
    print("加载预训练模型用于香蕉多果图伪标注...")
    try:
        pseudo_model = YOLO(PRETRAIN_MODEL)
    except Exception as e:
        print(f"  预训练模型加载失败（{e}），多果图将fallback为单果框")
        pseudo_model = None

    for split in SPLITS:
        img_root = Path(DATASET_ROOT) / "images" / split
        if not img_root.exists():
            print(f"[{split}] 目录不存在，跳过")
            continue
        print(f"\n── 处理 {split} ──")
        flatten_images(split)
        relabel(split, pseudo_model)

    print("\n✅ 重标注完成，请更新 data.yaml 后运行 dataset_check.py 验证")