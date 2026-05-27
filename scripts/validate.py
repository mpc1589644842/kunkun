from ultralytics import YOLO

if __name__ == '__main__':
    WEIGHT = "runs/detect/runs/detect/v2_16class/weights/best.pt"
    model  = YOLO(WEIGHT)

    metrics = model.val(
        data="fruit_dataset_v2/data.yaml",
        imgsz=640,
        device=0,
        split="val",
    )

    print("\n========== 验收结果 ==========")
    print(f"mAP@0.5:      {metrics.box.map50:.4f}   目标：>= 0.90")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}   目标：>= 0.80")
    print(f"Precision:    {metrics.box.mp:.4f}")
    print(f"Recall:       {metrics.box.mr:.4f}")

    print("\n========== 多目标场景测试 ==========")
    results = model.predict(
        source="fruit_dataset_v2/images/train/mosaic_0000.jpg",
        conf=0.25,
        iou=0.45,
        verbose=True,
        save=True,
        project="runs/detect",
        name="v2_multi_test",
    )
    boxes = results[0].boxes
    print(f"mosaic测试图检测到目标数：{len(boxes) if boxes else 0}  目标：>= 2")