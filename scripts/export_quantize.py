"""
export_quantize.py — 模型轻量化导出脚本
功能：
  1. 导出 ONNX FP32（基准）
  2. 导出 ONNX INT8（静态量化，需校准数据）
  3. 导出 TorchScript（PyTorch 原生加速）
  4. 性能对比报告（模型大小 / 推理时延 / mAP 估算）

运行方式（在 yolov11-fruit 环境下）：
  python export_quantize.py --weights runs/train/baseline/weights/best.pt
                            --data data.yaml
                            --calib-dir fruit_dataset/images/val
                            --imgsz 640
"""

import argparse
import time
import os
import sys
from pathlib import Path

import torch
import numpy as np
from ultralytics import YOLO


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────
def file_size_mb(path: str) -> float:
    return round(os.path.getsize(path) / 1024 / 1024, 2)


def benchmark_onnx(onnx_path: str, imgsz: int = 640, runs: int = 100) -> dict:
    """使用 onnxruntime 对 ONNX 模型做推理速度基准测试"""
    try:
        import onnxruntime as ort
    except ImportError:
        print("  [!] onnxruntime 未安装，跳过 ONNX 基准测试")
        print("      pip install onnxruntime-gpu  # GPU 版")
        print("      pip install onnxruntime       # CPU 版")
        return {}

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)
    inp_name = sess.get_inputs()[0].name

    dummy = np.random.rand(1, 3, imgsz, imgsz).astype(np.float32)

    # 预热
    for _ in range(10):
        sess.run(None, {inp_name: dummy})

    # 计时
    t0 = time.perf_counter()
    for _ in range(runs):
        sess.run(None, {inp_name: dummy})
    elapsed = (time.perf_counter() - t0) / runs * 1000  # ms

    return {
        "avg_latency_ms": round(elapsed, 2),
        "fps": round(1000 / elapsed, 1),
        "provider": sess.get_providers()[0],
    }


def benchmark_pt(model_path: str, imgsz: int = 640, runs: int = 100) -> dict:
    """对 PyTorch / TorchScript 模型做推理速度基准测试"""
    device = "cpu"   # 强制 CPU，避免 CUDA 设备 ID 冲突
    dummy = torch.rand(1, 3, imgsz, imgsz).to(device)

    if model_path.endswith(".torchscript"):
        model = torch.jit.load(model_path, map_location=device)
        model.eval()
        def infer(): return model(dummy)
    else:
        yolo = YOLO(model_path)
        yolo.to(device)
        def infer(): return yolo.predict(source=dummy, verbose=False, device="cpu")

    # 预热
    with torch.no_grad():
        for _ in range(5):
            infer()

    # 计时
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(runs):
            infer()
    elapsed = (time.perf_counter() - t0) / runs * 1000

    return {
        "avg_latency_ms": round(elapsed, 2),
        "fps": round(1000 / elapsed, 1),
        "device": device,
    }


# ─────────────────────────────────────────────
# 导出函数
# ─────────────────────────────────────────────
def export_onnx_fp32(weights: str, imgsz: int, output_dir: Path) -> str:
    """导出 ONNX FP32"""
    print("\n[1/3] 导出 ONNX FP32 ...")
    model = YOLO(weights)
    out = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=False,       # 固定 batch=1，便于量化
        simplify=True,       # onnx-simplifier 化简图结构
        opset=17,
    )
    dst = str(output_dir / "model_fp32.onnx")
    import shutil
    if Path(dst).exists():
        Path(dst).unlink()
    Path(out).rename(dst)
    print(f"  ✓ 已保存: {dst}  ({file_size_mb(dst)} MB)")
    return dst


def export_onnx_int8(fp32_onnx: str, calib_dir: str, imgsz: int, output_dir: Path) -> str:
    """
    ONNX INT8 静态量化
    依赖：onnxruntime, onnxruntime-extensions（可选）
          pip install onnxruntime-gpu onnx
    """
    print("\n[2/3] 导出 ONNX INT8 静态量化 ...")
    try:
        from onnxruntime.quantization import (
            quantize_static,
            CalibrationDataReader,
            QuantType,
            QuantFormat,
        )
        import onnx
        import cv2
    except ImportError as e:
        print(f"  [!] 缺少依赖: {e}")
        print("      pip install onnxruntime-gpu onnx opencv-python")
        return ""

    # ── 校准数据读取器 ──────────────────────────
    class FruitCalibReader(CalibrationDataReader):
        def __init__(self, calib_dir: str, imgsz: int, max_samples: int = 100):
            self.imgsz = imgsz
            self.data_iter = self._load(calib_dir, max_samples)

        def _load(self, calib_dir: str, max_samples: int):
            exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
            imgs = [
                p for p in Path(calib_dir).rglob("*")
                if p.suffix.lower() in exts
            ][:max_samples]

            if not imgs:
                raise FileNotFoundError(f"校准目录无图片: {calib_dir}")

            print(f"  使用 {len(imgs)} 张图片做 INT8 校准 ...")
            for p in imgs:
                img = cv2.imread(str(p))
                img = cv2.resize(img, (self.imgsz, self.imgsz))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0          # 归一化
                img = np.transpose(img, (2, 0, 1))[None]      # NCHW
                yield {"images": img}

        def get_next(self):
            return next(self.data_iter, None)

    dst = str(output_dir / "model_int8.onnx")
    quantize_static(
        model_input=fp32_onnx,
        model_output=dst,
        calibration_data_reader=FruitCalibReader(calib_dir, imgsz),
        quant_format=QuantFormat.QDQ,          # QDQ 格式兼容性最好
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
    )
    print(f"  ✓ 已保存: {dst}  ({file_size_mb(dst)} MB)")
    return dst


def export_torchscript(weights: str, imgsz: int, output_dir: Path) -> str:
    """导出 TorchScript（无需额外依赖，纯 PyTorch 加速）"""
    print("\n[3/3] 导出 TorchScript ...")
    model = YOLO(weights)
    out = model.export(format="torchscript", imgsz=imgsz, optimize=True)
    dst = str(output_dir / "model.torchscript")
    if Path(dst).exists():
        Path(dst).unlink()
    Path(out).rename(dst)
    print(f"  ✓ 已保存: {dst}  ({file_size_mb(dst)} MB)")
    return dst


# ─────────────────────────────────────────────
# 对比报告
# ─────────────────────────────────────────────
def print_report(weights, fp32_path, int8_path, ts_path, imgsz):
    print("\n" + "=" * 62)
    print("          模型轻量化对比报告")
    print("=" * 62)
    rows = []

    # 原始 PT
    r = benchmark_pt(weights, imgsz)
    rows.append({
        "model": "YOLOv11n (PT原始)",
        "size_mb": file_size_mb(weights),
        **r,
    })

    # ONNX FP32
    if fp32_path and Path(fp32_path).exists():
        r = benchmark_onnx(fp32_path, imgsz)
        rows.append({
            "model": "ONNX FP32",
            "size_mb": file_size_mb(fp32_path),
            **r,
        })

    # ONNX INT8
    if int8_path and Path(int8_path).exists():
        r = benchmark_onnx(int8_path, imgsz)
        rows.append({
            "model": "ONNX INT8 (量化)",
            "size_mb": file_size_mb(int8_path),
            **r,
        })

    # TorchScript
    if ts_path and Path(ts_path).exists():
        r = benchmark_pt(ts_path, imgsz)
        rows.append({
            "model": "TorchScript",
            "size_mb": file_size_mb(ts_path),
            **r,
        })

    print(f"{'模型':<22} {'大小(MB)':>8} {'时延(ms)':>10} {'FPS':>8}")
    print("-" * 52)
    baseline_size = rows[0]["size_mb"] if rows else 1
    for row in rows:
        compress = f"({row['size_mb']/baseline_size*100:.0f}%)"
        print(
            f"{row['model']:<22} "
            f"{row['size_mb']:>6.1f} {compress:>5}  "
            f"{row.get('avg_latency_ms', '-'):>8}  "
            f"{row.get('fps', '-'):>6}"
        )

    print("=" * 62)
    print("注：INT8 量化通常可在精度损失 < 1% 的前提下实现 2-4× 加速")
    print("    建议在 val 集上运行 model.val() 验证量化后 mAP@0.5\n")


# ─────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="水果检测模型轻量化导出工具")
    p.add_argument("--weights",
                   default="runs/detect/runs/detect/v2_16class/weights/best.pt",
                   help="训练好的 .pt 权重路径")
    p.add_argument("--data",      default="data.yaml",   help="data.yaml 路径")
    p.add_argument("--calib-dir", default="fruit_dataset/images/val",
                   help="INT8 量化校准图片目录（建议用 val 集）")
    p.add_argument("--imgsz",     type=int, default=640, help="输入分辨率")
    p.add_argument("--output",    default="exported_models",
                   help="导出目录")
    p.add_argument("--skip-int8", action="store_true",
                   help="跳过 INT8 量化（onnxruntime 未安装时使用）")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"权重文件 : {args.weights}")
    print(f"图像尺寸 : {args.imgsz}")
    print(f"校准目录 : {args.calib_dir}")
    print(f"输出目录 : {output_dir.resolve()}")

    fp32  = export_onnx_fp32(args.weights, args.imgsz, output_dir)
    int8  = "" if args.skip_int8 else export_onnx_int8(fp32, args.calib_dir, args.imgsz, output_dir)
    ts    = export_torchscript(args.weights, args.imgsz, output_dir)

    print_report(args.weights, fp32, int8, ts, args.imgsz)
    print("全部导出完成！")
