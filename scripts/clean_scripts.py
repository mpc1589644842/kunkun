#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_scripts.py  —  scripts 文件夹安全清理脚本

作用:把 scripts/ 目录下所有「开发过程中的一次性脚本」(以下划线 _ 开头的文件)
      移动到 scripts/_trash_待删除/ 文件夹,而不是直接删除。
      确认系统正常后,你再手动删掉 _trash_待删除 文件夹即可。

【为什么是移动而不是删除】
  移动是可逆的。万一误判,文件还在 _trash_待删除 里,随时能拿回来。
  等你跑过测试、确认系统正常,再删那个文件夹,就彻底安全了。

【使用方法】 在项目根目录(fruit_ripeness_std)下打开命令行,运行:

  第一步,预演(只看清单,不动任何文件):
      python scripts\\clean_scripts.py

  第二步,确认清单无误后,真正执行(加 --go 参数):
      python scripts\\clean_scripts.py --go

  第三步,执行后:
      1) 运行你的系统  python run_app.py  ,确认能正常启动
      2) 运行测试,确认 149 个测试仍然全部通过
      3) 一切正常后,手动删除 scripts\\_trash_待删除 整个文件夹
"""

import os
import sys
import shutil
from datetime import datetime

# ============ 配置区 ============

# 本脚本所在目录就是 scripts 文件夹
SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))

# 回收文件夹(放在 scripts 里面)
TRASH_DIR = os.path.join(SCRIPTS_DIR, "_trash_待删除")

# 【白名单】这些文件绝对不动,即使它以下划线开头也不动。
# 这是双保险:防止万一有重要文件碰巧以 _ 开头。
WHITELIST = {
    "clean_scripts.py",          # 本脚本自己
    # 测试文件(149 个单元测试在这里,绝对保留)
    "test_batch_repo.py",
    "test_state_machine.py",
    "test_supplier_repo.py",
    "test_user_repo.py",
    # 正经工具脚本(数据集 / 训练 / 导出)
    "auto_label.py", "auto_rename.py", "check_big_box.py", "cleanup.py",
    "compare_finetune_failure.py", "convert_webp.py", "dataset_check.py",
    "debug_raw.py", "export_quantize.py", "finetune.py", "merge_datasets.py",
    "mosaic_augment.py", "prepare_finetune_dataset.py", "quick_review.py",
    "relabel.py", "train.py", "validate.py", "val_int8.py",
}

# 【判定规则】文件名以这个前缀开头 → 判定为可清理的一次性脚本
TRASH_PREFIX = "_"


# ============ 主逻辑 ============

def is_trash(filename):
    """判断一个文件是否应被清理"""
    # 在白名单里 → 绝不清理
    if filename in WHITELIST:
        return False
    # 以下划线开头 → 判定为一次性脚本
    if filename.startswith(TRASH_PREFIX):
        return True
    return False


def main():
    # 是否真正执行(带 --go 才执行,否则只预演)
    really_do = "--go" in sys.argv

    print("=" * 60)
    print("  scripts 文件夹清理脚本")
    print("=" * 60)
    print(f"  目标目录: {SCRIPTS_DIR}")
    print(f"  模式:    {'★ 真正执行(移动文件)' if really_do else '预演(只看清单,不动文件)'}")
    print("=" * 60)

    # 收集所有要清理的文件
    trash_files = []
    keep_files = []

    for name in sorted(os.listdir(SCRIPTS_DIR)):
        full = os.path.join(SCRIPTS_DIR, name)
        # 跳过文件夹(包括回收文件夹自己)
        if os.path.isdir(full):
            continue
        if is_trash(name):
            trash_files.append(name)
        else:
            keep_files.append(name)

    # ---- 打印「保留」清单 ----
    print(f"\n【保留】以下 {len(keep_files)} 个文件不会被动:")
    for name in keep_files:
        print(f"    保留  {name}")

    # ---- 打印「清理」清单 ----
    print(f"\n【清理】以下 {len(trash_files)} 个文件将被移动到 _trash_待删除/ :")
    total_size = 0
    for name in trash_files:
        size = os.path.getsize(os.path.join(SCRIPTS_DIR, name))
        total_size += size
        print(f"    移动  {name}  ({size:,} 字节)")

    print("\n" + "-" * 60)
    print(f"  共 {len(trash_files)} 个文件,合计 {total_size:,} 字节 "
          f"(约 {total_size/1024:.0f} KB)")
    print("-" * 60)

    # ---- 预演模式:到此为止 ----
    if not really_do:
        print("\n[ 预演结束 ] 以上文件【尚未】被移动。")
        print("确认清单无误后,运行下面的命令真正执行:")
        print("    python scripts\\clean_scripts.py --go")
        return

    # ---- 执行模式:真正移动 ----
    if not trash_files:
        print("\n没有需要清理的文件,结束。")
        return

    # 创建回收文件夹
    os.makedirs(TRASH_DIR, exist_ok=True)

    # 在回收文件夹里写一个说明文件
    readme = os.path.join(TRASH_DIR, "_说明.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write("这个文件夹是 clean_scripts.py 在 "
                + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                + " 自动创建的。\n")
        f.write("里面是 scripts 文件夹清理出来的开发过程一次性脚本。\n\n")
        f.write("【确认系统正常后,可以整个删除这个文件夹】\n")
        f.write("确认方法:\n")
        f.write("  1. 运行 python run_app.py ,系统能正常启动\n")
        f.write("  2. 运行测试,149 个单元测试全部通过\n")
        f.write("  3. 以上都正常 → 删掉这个 _trash_待删除 文件夹\n")

    # 逐个移动
    moved = 0
    failed = []
    for name in trash_files:
        src = os.path.join(SCRIPTS_DIR, name)
        dst = os.path.join(TRASH_DIR, name)
        try:
            shutil.move(src, dst)
            moved += 1
        except Exception as e:
            failed.append((name, str(e)))

    print(f"\n[ 完成 ] 成功移动 {moved} 个文件到:")
    print(f"    {TRASH_DIR}")

    if failed:
        print(f"\n[ 注意 ] 有 {len(failed)} 个文件移动失败:")
        for name, err in failed:
            print(f"    {name}  ->  {err}")

    print("\n" + "=" * 60)
    print("  接下来请你做三件事:")
    print("  1. 运行  python run_app.py  ,确认系统能正常启动")
    print("  2. 运行测试,确认 149 个单元测试仍然全部通过")
    print("  3. 都正常后,手动删除整个 _trash_待删除 文件夹")
    print("=" * 60)
    print("\n  如果发现系统出问题,把 _trash_待删除 里的文件移回 scripts 即可恢复。")


if __name__ == "__main__":
    main()