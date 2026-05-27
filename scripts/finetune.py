"""
finetune.py — 富士苹果增量微调训练
Windows 多进程兼容版
"""
from ultralytics import YOLO
from pathlib import Path

# ── 路径配置 ──
START_WEIGHTS = "weights_backup/best_original.pt"
DATA_YAML     = "fuji_finetune/dataset/data.yaml"
PROJECT_DIR   = "runs/detect"
RUN_NAME      = "v2_finetune"

# ── 训练超参数 ──
EPOCHS    = 8
BATCH     = 8
IMGSZ     = 640
LR0       = 0.001
LRF       = 0.1
PATIENCE  = 3


def main():
    print("=" * 70)
    print("🚀 富士苹果 Finetune 启动")
    print("=" * 70)
    print(f"  起点权重: {START_WEIGHTS}")
    print(f"  数据配置: {DATA_YAML}")
    print(f"  输出目录: {PROJECT_DIR}/{RUN_NAME}")
    print(f"  Epoch:    {EPOCHS}")
    print(f"  Batch:    {BATCH}")
    print(f"  LR0:      {LR0} (原始 1/10)")
    print("=" * 70)

    # 检查起点权重
    if not Path(START_WEIGHTS).exists():
        print(f"❌ 起点权重不存在: {START_WEIGHTS}")
        print(f"   请先备份原始 best.pt 到 weights_backup/")
        return

    # 加载模型
    model = YOLO(START_WEIGHTS)

    # 训练
    import torch
    model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        batch=BATCH,
        imgsz=IMGSZ,
        lr0=LR0,
        lrf=LRF,
        patience=PATIENCE,

        # 输出目录
        project=PROJECT_DIR,
        name=RUN_NAME,
        exist_ok=True,

        # 数据增强(温和)
        mosaic=0.0,
        mixup=0.0,
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.4,
        fliplr=0.5,
        flipud=0.0,
        degrees=5,
        translate=0.1,
        scale=0.3,

        # 优化器与设备
        optimizer="AdamW",
        device="0" if torch.cuda.is_available() else "cpu",
        workers=0,
        seed=42,
        plots=True,
        save=True,
        val=True,
        verbose=True,
    )

    # 训练完成总结
    print("\n" + "=" * 70)
    print("✅ Finetune 完成")
    print("=" * 70)
    best_path = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    last_path = Path(PROJECT_DIR) / RUN_NAME / "weights" / "last.pt"
    print(f"  新权重(最佳):  {best_path}")
    print(f"  新权重(最后):  {last_path}")
    print(f"  原始权重未动:  weights_backup/best_original.pt")
    print("=" * 70)


if __name__ == "__main__":
    main()