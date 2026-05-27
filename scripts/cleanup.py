"""
cleanup.py — 同步删除 images/ 和 labels/ 中指定文件
"""
from pathlib import Path

IMG_DIR = Path("fuji_finetune/images")
LBL_DIR = Path("fuji_finetune/labels")

# ⚠️ 在这里维护要删除的文件名(不带后缀)
TO_DELETE = [
    "fuji_001",
    "fuji_008",
    "fuji_012",
    "fuji_013",
    "fuji_015",
    "fuji_019",
    "fuji_021",
    "fuji_034",
    "fuji_041",
    "fuji_042",
    "fuji_072",
]

deleted = 0
for stem in TO_DELETE:
    img_p = IMG_DIR / f"{stem}.jpg"
    lbl_p = LBL_DIR / f"{stem}.txt"
    if img_p.exists():
        img_p.unlink()
        deleted += 1
        print(f"  🗑️  删除 {img_p.name}")
    if lbl_p.exists():
        lbl_p.unlink()
        print(f"  🗑️  删除 {lbl_p.name}")

remaining = len(list(IMG_DIR.glob("*.jpg")))
print(f"\n✅ 删除 {deleted} 个图片  |  剩余 {remaining} 张")