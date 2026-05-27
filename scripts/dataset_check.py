"""
dataset_check.py
运行：python dataset_check.py
"""

from pathlib import Path
from collections import defaultdict

DATASET_ROOT = "./fruit_dataset_v2"
SPLITS       = ["train", "val", "test"]
CLASS_NAMES  = [
    "ripe_apple", "unripe_apple", "rotten_apple",
    "ripe_banana", "unripe_banana", "rotten_banana",
    "ripe_grape", "unripe_grape",
    "ripe_strawberry", "unripe_strawberry",
    "ripe_persimmon", "unripe_persimmon",
    "ripe_orange", "unripe_orange", "rotten_orange",
    "rotten_fruit",
]

def check_split(split):
    label_dir = Path(DATASET_ROOT) / "labels" / split
    image_dir = Path(DATASET_ROOT) / "images" / split

    if not label_dir.exists():
        print(f"[{split}] labels目录不存在，跳过")
        return None

    class_count    = defaultdict(int)
    multi_obj_imgs = 0
    multi_cls_imgs = 0
    error_files    = []
    total_imgs     = 0

    txt_files = list(label_dir.glob("*.txt"))
    # 兼容按类别分子目录的结构
    for sub in label_dir.iterdir():
        if sub.is_dir():
            txt_files += list(sub.glob("*.txt"))

    for txt in txt_files:
        total_imgs += 1
        lines = txt.read_text(encoding="utf-8").strip().splitlines()
        lines = [l for l in lines if l.strip()]

        if not lines:
            error_files.append((txt.name, "空标注文件"))
            continue

        cls_in_img = set()
        for i, line in enumerate(lines):
            parts = line.strip().split()
            if len(parts) != 5:
                error_files.append((txt.name, f"第{i+1}行字段数={len(parts)}，应为5"))
                continue
            try:
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
            except ValueError:
                error_files.append((txt.name, f"第{i+1}行数值解析失败"))
                continue

            if not (0 <= cls_id <= 15):   # 原 <= 7，改为 <= 15
                error_files.append((txt.name, f"class_id={cls_id}越界"))
                continue
            if not all(0.0 <= v <= 1.0 for v in [cx, cy, w, h]):
                error_files.append((txt.name, f"坐标越界:{cx:.3f},{cy:.3f},{w:.3f},{h:.3f}"))
                continue
            if w < 0.01 or h < 0.01:
                error_files.append((txt.name, f"框过小:w={w:.4f} h={h:.4f}"))

            class_count[cls_id] += 1
            cls_in_img.add(cls_id)

        if len(lines) > 1:
            multi_obj_imgs += 1
        if len(cls_in_img) > 1:
            multi_cls_imgs += 1

    print(f"\n{'='*55}")
    print(f"[{split}]  标注文件共 {total_imgs} 个")
    print(f"  多目标图（≥2个框）：{multi_obj_imgs} 个  ({multi_obj_imgs/max(total_imgs,1)*100:.1f}%)")
    print(f"  多品类混合图：      {multi_cls_imgs} 个  ({multi_cls_imgs/max(total_imgs,1)*100:.1f}%)")
    print(f"\n  各类别标注框数：")
    max_cnt = max(class_count.values()) if class_count else 1
    for cid, name in enumerate(CLASS_NAMES):
        cnt = class_count[cid]
        bar = "█" * int(cnt / max(max_cnt, 1) * 30)
        print(f"    {cid}  {name:<18} {cnt:>5}  {bar}")

    if error_files:
        print(f"\n  ⚠️  发现 {len(error_files)} 个标注问题（最多显示15条）：")
        for fname, reason in error_files[:15]:
            print(f"    {fname}: {reason}")
    else:
        print(f"\n  ✅ 无标注异常")

    return {
        "total":       total_imgs,
        "multi_obj":   multi_obj_imgs,
        "multi_cls":   multi_cls_imgs,
        "class_count": dict(class_count),
        "errors":      error_files,
    }

if __name__ == "__main__":
    for split in SPLITS:
        stats = check_split(split)

    print(f"\n{'='*55}")
    print("运行完成，把以上输出截图发给我")