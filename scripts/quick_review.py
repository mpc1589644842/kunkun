"""
quick_review.py — 把 fuji_finetune/images/ 下的 79 张图拼成一个网格大图
便于一眼扫过去,挑出问题图(变色/水印/过度修图)
输出:fuji_finetune/_review_grid.jpg
"""
from pathlib import Path
import math
from PIL import Image, ImageDraw, ImageFont

IMG_DIR = Path("fuji_finetune/images")
OUT_PATH = Path("fuji_finetune/_review_grid.jpg")

THUMB_SIZE = 200          # 每张缩略图大小
COLS = 8                   # 每行 8 张
PADDING = 4                # 缩略图间距
LABEL_HEIGHT = 22          # 文件名标签高度

imgs = sorted(IMG_DIR.glob("*.jpg"))
n = len(imgs)
rows = math.ceil(n / COLS)

# 总画布尺寸
W = COLS * (THUMB_SIZE + PADDING) + PADDING
H = rows * (THUMB_SIZE + LABEL_HEIGHT + PADDING) + PADDING

canvas = Image.new("RGB", (W, H), (240, 240, 240))
draw = ImageDraw.Draw(canvas)

# 加载字体(用于标文件名)
font = None
for fc in ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/msyh.ttc"]:
    try:
        font = ImageFont.truetype(fc, 13)
        break
    except Exception:
        continue
if font is None:
    font = ImageFont.load_default()

print(f"📋 找到 {n} 张图,生成 {COLS}×{rows} 网格预览...")

for i, p in enumerate(imgs):
    row = i // COLS
    col = i % COLS
    x = PADDING + col * (THUMB_SIZE + PADDING)
    y = PADDING + row * (THUMB_SIZE + LABEL_HEIGHT + PADDING)

    # 缩略图
    img = Image.open(p).convert("RGB")
    img.thumbnail((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)
    # 居中粘贴
    paste_x = x + (THUMB_SIZE - img.width) // 2
    paste_y = y + (THUMB_SIZE - img.height) // 2
    canvas.paste(img, (paste_x, paste_y))

    # 文件名
    label_y = y + THUMB_SIZE + 2
    draw.text((x + 4, label_y), p.stem, fill=(50, 50, 50), font=font)

canvas.save(OUT_PATH, "JPEG", quality=88)
print(f"\n✅ 完成 → {OUT_PATH}")
print(f"   画布尺寸: {W} × {H}")
print(f"   用图片查看器打开,逐行扫一遍即可")