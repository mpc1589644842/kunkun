import shutil
import random
from pathlib import Path

NEW_DATASET = r"C:\Users\kun\Desktop\fruit_ripeness_std\fruit_dataset\datasets"
OLD_DATASET = "./fruit_dataset"
OUT_DATASET = "./fruit_dataset_v2"
SEED = 42
random.seed(SEED)

FINAL_CLASS_NAMES = [
    "ripe_apple", "unripe_apple", "rotten_apple",
    "ripe_banana", "unripe_banana", "rotten_banana",
    "ripe_grape", "unripe_grape",
    "ripe_strawberry", "unripe_strawberry",
    "ripe_persimmon", "unripe_persimmon",
    "ripe_orange", "unripe_orange", "rotten_orange",
    "rotten_fruit",
]

NEW_CLASS_MAP_FINAL = {
    0:  0,
    1:  0,
    2:  1,
    3:  3,
    4:  4,
    5:  4,
    6:  6,
    7:  7,
    8:  7,
    9:  8,
    10: 9,
    11: 9,
    12: 10,
    13: 11,
    14: 11,
    15: 15,
}

OLD_CLASS_MAP_FINAL = {
    0: 0,
    1: 2,
    2: 3,
    3: 5,
    4: 4,
    5: 12,
    6: 14,
    7: 13,
}


def remap_label_file(src, dst, class_map):
    lines_out = []
    for line in Path(src).read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        old_id = int(parts[0])
        if old_id not in class_map:
            continue
        new_id = class_map[old_id]
        lines_out.append(f"{new_id} {' '.join(parts[1:])}")
    if lines_out:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_text("\n".join(lines_out))
        return True
    return False


def copy_split(src_img_dir, src_lbl_dir, out_img_dir, out_lbl_dir, class_map, prefix=""):
    copied = 0
    skipped = 0
    for img in Path(src_img_dir).glob("*.[jp][pn]g"):
        lbl = Path(src_lbl_dir) / (img.stem + ".txt")
        if not lbl.exists():
            skipped += 1
            continue
        new_name = f"{prefix}{img.name}" if prefix else img.name
        dst_img = Path(out_img_dir) / new_name
        dst_lbl = Path(out_lbl_dir) / f"{prefix}{img.stem}.txt"
        if remap_label_file(lbl, dst_lbl, class_map):
            shutil.copy2(img, dst_img)
            copied += 1
        else:
            skipped += 1
    return copied, skipped


def copy_new_split(src_img_dir, src_lbl_dir, out_img_dir, out_lbl_dir, rotten_limit):
    copied = 0
    skipped = 0
    rotten_count = 0
    for img in Path(src_img_dir).glob("*.[jp][pn]g"):
        lbl = Path(src_lbl_dir) / (img.stem + ".txt")
        if not lbl.exists():
            skipped += 1
            continue
        lines = lbl.read_text().strip().splitlines()
        cls_ids = set()
        for line in lines:
            parts = line.strip().split()
            if len(parts) == 5:
                old_id = int(parts[0])
                new_id = NEW_CLASS_MAP_FINAL.get(old_id, -1)
                if new_id >= 0:
                    cls_ids.add(new_id)
        if cls_ids == {15}:
            if rotten_count >= rotten_limit:
                skipped += 1
                continue
            rotten_count += 1
        dst_img = Path(out_img_dir) / f"new_{img.name}"
        dst_lbl = Path(out_lbl_dir) / f"new_{img.stem}.txt"
        if remap_label_file(lbl, dst_lbl, NEW_CLASS_MAP_FINAL):
            shutil.copy2(img, dst_img)
            copied += 1
        else:
            skipped += 1
    print(f"  新数据集：复制{copied}张，跳过{skipped}张，rotten_fruit限制{rotten_count}张")
    return copied


if __name__ == "__main__":
    out = Path(OUT_DATASET)
    if out.exists():
        shutil.rmtree(out)

    for split in ["train", "val"]:
        out_img = out / "images" / split
        out_lbl = out / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        print(f"\n── {split} ──")

        rotten_limit = 500 if split == "train" else 100
        new_img = Path(NEW_DATASET) / split / "images"
        new_lbl = Path(NEW_DATASET) / split / "labels"
        if new_img.exists():
            copy_new_split(new_img, new_lbl, out_img, out_lbl, rotten_limit)
        else:
            print(f"  新数据集 {split} 目录不存在，跳过")

        old_img = Path(OLD_DATASET) / "images" / split
        old_lbl = Path(OLD_DATASET) / "labels" / split
        if old_img.exists():
            c, s = copy_split(old_img, old_lbl, out_img, out_lbl,
                              OLD_CLASS_MAP_FINAL, prefix="old_")
            print(f"  现有数据集：复制{c}张，跳过{s}张")
        else:
            print(f"  现有数据集 {split} 目录不存在，跳过")

    yaml_content = f"path: {out.resolve()}\n\ntrain: images/train\nval:   images/val\n\nnc: {len(FINAL_CLASS_NAMES)}\nnames:\n"
    for i, name in enumerate(FINAL_CLASS_NAMES):
        yaml_content += f"  {i}: {name}\n"
    yaml_out = out / "data.yaml"
    yaml_out.write_text(yaml_content)
    print(f"\n✅ data.yaml 已生成：{yaml_out}")
    print("运行 python dataset_check.py 验证合并结果")