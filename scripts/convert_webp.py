"""
convert_webp.py — 把 fuji_finetune/images_raw/ 下的 webp 转成 jpg
原 webp 文件保留(以防万一),新 jpg 输出到 images/ 目录
"""
from pathlib import Path
from PIL import Image

src_dir = Path("fuji_finetune/images_raw")
dst_dir = Path("fuji_finetune/images")
dst_dir.mkdir(parents=True, exist_ok=True)

webps = sorted(src_dir.glob("*.webp"))
print(f"找到 {len(webps)} 张 webp")

for i, p in enumerate(webps, 1):
    img = Image.open(p).convert("RGB")
    new_name = f"fuji_{i:03d}.jpg"
    img.save(dst_dir / new_name, "JPEG", quality=95)
    print(f"  [{i}/{len(webps)}] {p.name} → {new_name}")

print(f"\n✅ 完成,输出到 {dst_dir}")