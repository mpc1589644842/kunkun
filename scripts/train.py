from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo11n.pt")

    model.train(
        data="fruit_dataset_v2/data.yaml",
        epochs=150,
        imgsz=640,
        batch=16,
        device=0,
        patience=30,
        cos_lr=True,
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        mosaic=0.5,
        mixup=0.1,
        copy_paste=0.1,
        degrees=10.0,
        fliplr=0.5,
        scale=0.4,
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,
        project=".",
        name="runs/detect/v2_16class",
        save=True,
        plots=True,
    )