"""
debug_raw.py — 诊断脚本：跳过所有业务过滤，直接看 YOLO 原始输出
用法：python debug_raw.py <图片路径>
示例：python debug_raw.py test_apple.webp
"""

import sys
from pathlib import Path
from ultralytics import YOLO

# ── 配置 ──
MODEL_PATH = "runs/detect/runs/detect/v2_16class/weights/best.pt"
CLASS_NAMES = [
    "ripe_apple", "unripe_apple", "rotten_apple",
    "ripe_banana", "unripe_banana", "rotten_banana",
    "ripe_grape", "unripe_grape",
    "ripe_strawberry", "unripe_strawberry",
    "ripe_persimmon", "unripe_persimmon",
    "ripe_orange", "unripe_orange", "rotten_orange",
    "rotten_fruit",
]

# ── 解析参数 ──
if len(sys.argv) < 2:
    print("用法: python debug_raw.py <图片路径>")
    sys.exit(1)

img_path = sys.argv[1]
if not Path(img_path).exists():
    print(f"❌ 图片不存在: {img_path}")
    sys.exit(1)

# ── 加载模型 ──
print(f"📦 加载模型: {MODEL_PATH}")
model = YOLO(MODEL_PATH)

# ── 三轮推理：从极低阈值到正常阈值，看模型到底"看到"了什么 ──
print(f"\n🖼  图片: {img_path}")
print("=" * 75)

for conf_thres in [0.05, 0.25, 0.50]:
    print(f"\n▶ conf_thres = {conf_thres}")
    print("-" * 75)
    results = model.predict(
        source=img_path,
        conf=conf_thres,
        iou=0.45,
        verbose=False,
        device="cpu",
    )

    if not results or results[0].boxes is None or len(results[0].boxes) == 0:
        print("  (无任何检测框)")
        continue

    boxes = results[0].boxes
    print(f"  共 {len(boxes)} 个框:")
    print(f"  {'#':<3} {'类别':<22} {'置信度':<8} {'坐标 (x1,y1,x2,y2)'}")
    for i, box in enumerate(boxes):
        cid  = int(box.cls.item())
        conf = float(box.conf.item())
        x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
        cname = CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else f"unknown({cid})"
        print(f"  {i+1:<3} {cname:<22} {conf:.4f}   ({x1},{y1},{x2},{y2})")

print("\n" + "=" * 75)
print("✅ 诊断完成")