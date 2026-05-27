"""
val_int8.py — 用 ONNXRuntime 直接验证 INT8 量化模型精度
正确方式：绕过 ultralytics 的 ONNX 解析，直接用 onnxruntime 推理
"""

import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path
from tqdm import tqdm
import yaml

# ── 配置 ──────────────────────────────────────
ONNX_PATH  = "exported_models/model_int8.onnx"
DATA_YAML  = "data.yaml"
CONF_THRES = 0.25
IOU_THRES  = 0.45
IMGSZ      = 640

# ── 加载数据集路径 ─────────────────────────────
with open(DATA_YAML, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

dataset_root = Path(cfg["path"])
val_img_dir  = dataset_root / cfg["val"]
val_lbl_dir  = str(val_img_dir).replace("images", "labels")
class_names  = list(cfg["names"].values())
nc           = cfg["nc"]

# ── 加载模型 ───────────────────────────────────
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
sess      = ort.InferenceSession(ONNX_PATH, providers=providers)
inp_name  = sess.get_inputs()[0].name
print(f"模型加载成功，使用: {sess.get_providers()[0]}")
print(f"输入节点: {inp_name}, shape: {sess.get_inputs()[0].shape}")

# ── 预处理 ────────────────────────────────────
def preprocess(img_path):
    img = cv2.imread(str(img_path))
    h0, w0 = img.shape[:2]
    img_resized = cv2.resize(img, (IMGSZ, IMGSZ))
    img_rgb     = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    blob        = img_rgb.astype(np.float32) / 255.0
    blob        = np.transpose(blob, (2, 0, 1))[None]  # NCHW
    return blob, (h0, w0)

# ── NMS ───────────────────────────────────────
def nms(boxes, scores, iou_thres):
    x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    areas = (x2-x1)*(y2-y1)
    order = scores.argsort()[::-1]
    keep  = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w   = np.maximum(0, xx2-xx1)
        h   = np.maximum(0, yy2-yy1)
        iou = (w*h) / (areas[i]+areas[order[1:]]-w*h+1e-6)
        order = order[np.where(iou<=iou_thres)[0]+1]
    return keep

# ── 后处理（YOLOv11 输出格式：[1, 12, 8400]）──
def postprocess(output, conf_thres, orig_hw):
    # output shape: (1, 4+nc, 8400)
    pred   = output[0][0]              # (12, 8400)
    pred   = pred.T                    # (8400, 12)
    boxes  = pred[:, :4]               # cx,cy,w,h
    scores = pred[:, 4:]               # (8400, nc)

    scores = 1 / (1 + np.exp(-np.clip(scores, -500, 500)))
    # 转 xyxy
    cx,cy,w,h = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
    x1 = (cx - w/2) / IMGSZ
    y1 = (cy - h/2) / IMGSZ
    x2 = (cx + w/2) / IMGSZ
    y2 = (cy + h/2) / IMGSZ
    boxes_xyxy = np.stack([x1,y1,x2,y2], axis=1)

    cls_ids    = np.argmax(scores, axis=1)
    cls_confs  = np.max(scores, axis=1)
    mask       = cls_confs > conf_thres

    boxes_f  = boxes_xyxy[mask]
    confs_f  = cls_confs[mask]
    cls_f    = cls_ids[mask]

    results = []
    for c in range(nc):
        cidx = np.where(cls_f == c)[0]
        if len(cidx) == 0: continue
        kept = nms(boxes_f[cidx], confs_f[cidx], IOU_THRES)
        for k in kept:
            results.append({
                "box":  boxes_f[cidx[k]].tolist(),
                "conf": float(confs_f[cidx[k]]),
                "cls":  c,
            })
    return results

# ── 读取标签 ──────────────────────────────────
def load_labels(lbl_path):
    if not Path(lbl_path).exists(): return []
    labels = []
    with open(lbl_path) as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            cls = int(parts[0])
            cx,cy,w,h = parts[1],parts[2],parts[3],parts[4]
            labels.append({"cls": cls,
                           "box": [cx-w/2, cy-h/2, cx+w/2, cy+h/2]})
    return labels

# ── IoU ───────────────────────────────────────
def box_iou(b1, b2):
    xi1=max(b1[0],b2[0]); yi1=max(b1[1],b2[1])
    xi2=min(b1[2],b2[2]); yi2=min(b1[3],b2[3])
    inter=max(0,xi2-xi1)*max(0,yi2-yi1)
    a1=(b1[2]-b1[0])*(b1[3]-b1[1])
    a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    return inter/(a1+a2-inter+1e-6)

# ── 主评估循环 ────────────────────────────────
img_paths = sorted(Path(val_img_dir).rglob("*.jpg")) + \
            sorted(Path(val_img_dir).rglob("*.png"))

print(f"\n验证集图片数: {len(img_paths)}")

tp_all = np.zeros(nc)
fp_all = np.zeros(nc)
fn_all = np.zeros(nc)

for img_path in tqdm(img_paths, desc="验证中"):
    blob, orig_hw = preprocess(img_path)
    output  = sess.run(None, {inp_name: blob})
    preds   = postprocess(output, CONF_THRES, orig_hw)

    lbl_path = Path(val_lbl_dir) / (img_path.stem + ".txt")
    labels   = load_labels(lbl_path)

    matched_gt = set()
    for pred in preds:
        best_iou, best_gt = 0, -1
        for gi, gt in enumerate(labels):
            if gt["cls"] != pred["cls"]: continue
            iou = box_iou(pred["box"], gt["box"])
            if iou > best_iou:
                best_iou, best_gt = iou, gi
        if best_iou >= 0.5 and best_gt not in matched_gt:
            tp_all[pred["cls"]] += 1
            matched_gt.add(best_gt)
        else:
            fp_all[pred["cls"]] += 1

    for gi, gt in enumerate(labels):
        if gi not in matched_gt:
            fn_all[gt["cls"]] += 1

# ── 结果输出 ──────────────────────────────────
print("\n" + "="*55)
print("  INT8 量化模型验证结果 (IoU=0.5)")
print("="*55)
print(f"{'类别':<18} {'P':>6} {'R':>6} {'F1':>6}")
print("-"*40)

all_p, all_r = [], []
for c, name in enumerate(class_names):
    tp = tp_all[c]; fp = fp_all[c]; fn = fn_all[c]
    p  = tp/(tp+fp+1e-6)
    r  = tp/(tp+fn+1e-6)
    f1 = 2*p*r/(p+r+1e-6)
    all_p.append(p); all_r.append(r)
    print(f"{name:<18} {p:>6.3f} {r:>6.3f} {f1:>6.3f}")

mp  = np.mean(all_p)
mr  = np.mean(all_r)
mf1 = 2*mp*mr/(mp+mr+1e-6)
print("-"*40)
print(f"{'全部类别 (mean)':<18} {mp:>6.3f} {mr:>6.3f} {mf1:>6.3f}")
print(f"\n⚠️  注：此脚本计算的是 P/R/F1，完整 mAP 请用原始PT模型对比")
print("="*55)