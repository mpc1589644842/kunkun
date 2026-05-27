"""
_reorganize2.py — 第二轮整理:处理剩余文件
策略:
  - 工具脚本 → scripts/
  - 不确定的文件 → _archive/(隔离区,留几天观察)
  - 测试图 → paper_assets/test_images/
  - 预训练权重 → weights_backup/
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
SCRIPTS_DIR = ROOT / "scripts"
ARCHIVE_DIR = ROOT / "_archive"      # 不确定文件先扔这,确认无用后再删
TEST_IMG_DIR = ROOT / "paper_assets" / "test_images"
WEIGHTS_DIR = ROOT / "weights_backup"

ARCHIVE_DIR.mkdir(exist_ok=True)
TEST_IMG_DIR.mkdir(parents=True, exist_ok=True)

# ── 工具脚本 → scripts/ ──
TO_SCRIPTS = ["relabel.py", "val_int8.py"]

# ── 测试图 → paper_assets/test_images/ ──
TO_TEST_IMG = ["apple_test.webp"]

# ── 预训练权重 → weights_backup/ ──
TO_WEIGHTS = ["yolo11n.pt"]

# ── 不确定文件 → _archive/(隔离观察)──
TO_ARCHIVE = ["yolo26n.pt", "setup_and_run.py"]


def safe_move(fname, dst_dir, label):
    src = ROOT / fname
    if src.exists():
        dst = dst_dir / fname
        shutil.move(str(src), str(dst))
        print(f"  {label} {fname} → {dst_dir.name}/")
        return 1
    else:
        print(f"  ⏭️  {fname} 不存在,跳过")
        return 0


print("=" * 60)
print("📦 开始第二轮整理")
print("=" * 60)

count = 0
for f in TO_SCRIPTS:    count += safe_move(f, SCRIPTS_DIR,   "🔧")
for f in TO_TEST_IMG:   count += safe_move(f, TEST_IMG_DIR,  "🖼️")
for f in TO_WEIGHTS:    count += safe_move(f, WEIGHTS_DIR,   "⚖️")
for f in TO_ARCHIVE:    count += safe_move(f, ARCHIVE_DIR,   "📦")

print(f"\n✅ 完成,共移动 {count} 个文件")

print(f"\n🔍 根目录现状:")
for p in sorted(ROOT.iterdir()):
    if p.name.startswith(".") or p.name.startswith("_") and p.is_file():
        continue
    icon = "📁" if p.is_dir() else "📄"
    print(f"   {icon} {p.name}")