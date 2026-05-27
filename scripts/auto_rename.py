from pathlib import Path

# 8 类顺序（必须和 data.yaml 完全一致）
class_order = [
    "ripe_apple",
    "rotten_apple",
    "ripe_banana",
    "rotten_banana",
    "unripe_banana",
    "ripe_orange",
    "rotten_orange",
    "unripe_orange",
]

root = Path("fruit_dataset")

for split in ["train", "val", "test"]:
    img_dir = root / "images" / split

    # 遍历每个类别
    for cls_name in class_order:
        cls_folder = img_dir / cls_name
        if not cls_folder.exists():
            print(f"⚠️  未找到目录: {cls_folder}")
            continue

        # 获取该类别下所有图片
        img_paths = list(cls_folder.glob("*.jpg")) + list(cls_folder.glob("*.png"))
        if not img_paths:
            continue

        # 按序号重命名 (ripe_apple_0001.jpg)
        for idx, img_path in enumerate(img_paths, 1):
            # 格式化序号为 4 位 (0001)
            new_name = f"{cls_name}_{idx:04d}{img_path.suffix}"
            new_path = cls_folder / new_name

            # 执行重命名
            img_path.rename(new_path)
            print(f"✅ 已重命名: {img_path.name} -> {new_name}")

print("\n🎉 所有图片重命名完成！")