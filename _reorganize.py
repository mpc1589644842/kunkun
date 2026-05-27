"""
_reorganize.py — 一次性整理项目结构
将工具脚本移到 scripts/,核心业务文件保留根目录
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).parent
SCRIPTS_DIR = ROOT / "scripts"
SCRIPTS_DIR.mkdir(exist_ok=True)

# 移到 scripts/ 的工具脚本(根据你目录截图)
TO_MOVE = [
    "auto_label.py",
    "auto_rename.py",
    "check_big_box.py",
    "cleanup.py",
    "compare_finetune_failure.py",
    "convert_webp.py",
    "dataset_check.py",
    "debug_raw.py",
    "export_quantize.py",
    "finetune.py",
    "merge_datasets.py",
    "mosaic_augment.py",
    "prepare_finetune_dataset.py",
    "quick_review.py",
    "train.py",
    "validate.py",
]

# 保留根目录(主流程)
KEEP_IN_ROOT = ["app.py", "grading.py", "report_generator.py", "data.yaml"]

moved = 0
for fname in TO_MOVE:
    src = ROOT / fname
    if src.exists():
        dst = SCRIPTS_DIR / fname
        shutil.move(str(src), str(dst))
        print(f"  📦 {fname} → scripts/")
        moved += 1
    else:
        print(f"  ⏭️  {fname} 不存在,跳过")

print(f"\n✅ 移动完成,共 {moved} 个文件")
print(f"\n🔍 根目录现状:")
for p in sorted(ROOT.iterdir()):
    if p.name.startswith("_") or p.name.startswith("."):
        continue
    icon = "📁" if p.is_dir() else "📄"
    print(f"   {icon} {p.name}")