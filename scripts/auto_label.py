"""
auto_label.py — 用 best.pt 对 fuji_finetune/images 自动预标注
输出 YOLO 格式 .txt 到 fuji_finetune/labels/
强制所有 class_id = 0 (ripe_apple)
"""
from pathlib import Path
import cv2
from ultralytics import YOLO

MODEL_PATH = "runs/detect/runs/detect/v2_16class/weights/best.pt"
IMG_DIR    = Path("fuji_finetune/images")
LBL_DIR    = Path("fuji_finetune/labels")
LBL_DIR.mkdir(parents=True, exist_ok=True)

# 极低置信度,确保模型把所有"看起来像水果"的目标都框出来
CONF_THRES = 0.10
IOU_THRES  = 0.45
TARGET_CLASS = 0  # 强制全部改为 ripe_apple

print(f"📦 加载模型: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

imgs = sorted(IMG_DIR.glob("*.jpg"))
print(f"🖼  找到 {len(imgs)} 张图\n")

success_count = 0
empty_count   = 0

for i, img_path in enumerate(imgs, 1):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [{i}/{len(imgs)}] ❌ 无法读取: {img_path.name}")
        continue
    h, w = img.shape[:2]

    results = model.predict(source=img, conf=CONF_THRES, iou=IOU_THRES,
                            verbose=False, device="cpu")

    boxes = results[0].boxes
    txt_lines = []
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            # YOLO 格式:class cx_norm cy_norm w_norm h_norm
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            # 强制类别为 0 (ripe_apple)
            txt_lines.append(f"{TARGET_CLASS} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

    txt_path = LBL_DIR / (img_path.stem + ".txt")
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")

    if txt_lines:
        success_count += 1
        print(f"  [{i}/{len(imgs)}] ✅ {img_path.name} → {len(txt_lines)} 个框")
    else:
        empty_count += 1
        print(f"  [{i}/{len(imgs)}] ⚠️ {img_path.name} → 0 个框 (需手工标注)")

print(f"\n{'='*60}")
print(f"✅ 完成 — 成功: {success_count} 张  |  空标签: {empty_count} 张")
print(f"📁 标签输出: {LBL_DIR}")
print(f"{'='*60}")