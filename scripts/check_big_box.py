"""
check_big_box.py — 找出标签里"框过大"的图,这些通常是群体框误标
"""
from pathlib import Path

LBL_DIR = Path("fuji_finetune/labels")
IMG_DIR = Path("fuji_finetune/images")

# 检查阈值:框面积 > 整图 60% 视为可疑
SUSPICIOUS_AREA_RATIO = 0.60

print(f"🔍 扫描 {LBL_DIR}\n")

suspicious = []
empty = []

for lbl_p in sorted(LBL_DIR.glob("*.txt")):
    if lbl_p.name == "classes.txt":
        continue
    lines = lbl_p.read_text(encoding="utf-8").strip().split("\n")
    lines = [l for l in lines if l.strip()]

    if not lines:
        empty.append(lbl_p.stem)
        continue

    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        w_norm = float(parts[3])
        h_norm = float(parts[4])
        area = w_norm * h_norm
        if area > SUSPICIOUS_AREA_RATIO:
            suspicious.append((lbl_p.stem, area, len(lines)))
            break

print(f"⚠️  发现 {len(suspicious)} 张可疑(框>60%整图):")
for stem, area, n_box in suspicious:
    print(f"   {stem}.jpg  框占比={area*100:.0f}%  共{n_box}个框")

if empty:
    print(f"\n📭 空标签 {len(empty)} 张(需手工补框):")
    for stem in empty:
        print(f"   {stem}.jpg")

print(f"\n✅ 完成。建议把可疑图在 LabelImg 里打开核对,其他图可以跳过核对直接进训练。")