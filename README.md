# 基于轻量化 YOLOv11 的水果入库盘点与品质分级系统

> **Lightweight YOLOv11-Based Fruit Inbound Reconciliation and Quality Grading System**

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.2-red)](https://pytorch.org/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.31-orange)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)



---

## 📋 目录

- [项目背景](#项目背景)
- [核心特点](#核心特点)
- [系统架构](#系统架构)
- [模型性能](#模型性能)
- [功能模块](#功能模块)
- [快速开始](#快速开始)
- [数据集](#数据集)
- [项目结构](#项目结构)
- [单元测试](#单元测试)
- [许可证](#许可证)

---

## 项目背景

水果入库环节的盘点与品质判断长期依赖人工目检，存在效率低、标准不统一、缺乏数字化追溯等痛点。工业级分选设备价格高昂（数十万至上百万元），且多为单一品种流水线设计，与中小经销商"多品种、小批次、高周转"的实际运营模式不匹配。

本系统以**轻量化 YOLOv11n** 为核心检测模型，面向中小水果经销商，构建了一套"模型自动判断 → 规则硬约束 → 人工复核兜底"三位一体的可信入库工作流，在普通笔记本电脑上即可稳定运行。

---

## 核心特点

### 🧠 模型层
- **YOLOv11n** 作为骨干网络，2.6M 参数，6.5G FLOPs
- 验证集 **mAP@0.5 = 0.971**，RTX 3050 Laptop 上 **35.7 FPS** 实时推理
- 16 类细粒度水果状态检测，覆盖苹果、香蕉、葡萄、橙子四大品类
- 支持 ONNX / TorchScript / INT8 量化导出

### ⚙️ 工程层 — "四层轻量化设计"框架
| 层级 | 策略 | 效果 |
|------|------|------|
| **模型选型层** | YOLOv11n vs s/m 对比，选最小可用模型 | 参数减少 2.6-7.7×，精度损失仅 0.4-0.7% |
| **部署形式层** | ONNX Runtime 替代 PyTorch 原生推理 | 部署体积压缩 **~97%**（2GB → 50MB） |
| **算法应用层** | 16→3 类别映射 + 跨类 NMS + 置信度阈值 | 不改模型结构，消除误检冲突 |
| **数据策略层** | 三阶段递进式数据集构建 | 人工标注成本压缩 **~95%** |

### 🔄 业务层
- **五状态批次状态机**：草稿 → 待确认 → 待复核 → 已入库 / 已拒收
- **三阈值快速通道**：数量差异 <3%、次品率 <5%、检测数 ≥10，全满足自动放行
- **三位一体可信流程**：模型 + 规则 + 人工协同决策
- **复核 KPI 快照**：经理复核时冻结当前指标，决策留痕可追溯

### 🏗️ 数据持久化
- SQLAlchemy 2.0 ORM + SQLite，零配置部署
- Repository 模式解耦业务与数据访问
- bcrypt 哈希存储用户密码，验证后立即抹除明文
- 操作日志只追加不修改，完整审计链

### 👥 多角色协作
| 角色 | 权限 |
|------|------|
| **操作员** | 创建批次、执行检测、确认快速通道入库 |
| **经理** | 复核异常批次（通过/拒收）、查看全部历史 |
| **管理员** | 维护供应商、水果品类、用户账号 |

---

## 系统架构

```
┌──────────────────────────────────────────────┐
│              用户界面层 (Streamlit)              │
│  登录页 │ 检测台 │ 批次管理 │ 复核中心 │ 供应商管理 │  ...  │
├──────────────────────────────────────────────┤
│              业务逻辑层 (services/)              │
│  状态机 │ 三阈值判定 │ ABC分级 │ 跨类NMS │ RBAC权限  │
├──────────────────────────────────────────────┤
│            数据持久化层 (SQLAlchemy + SQLite)      │
│  Repository 模式 │ ORM 映射 │ bcrypt 密码哈希     │
├──────────────────────────────────────────────┤
│           模型推理层 (ONNX Runtime)              │
│  YOLOv11n → ONNX Export → ONNX Runtime 推理    │
└──────────────────────────────────────────────┘
```

---

## 模型性能

### YOLOv11 系列对比（同一数据集）

| 模型 | 参数量 | mAP@0.5 | FPS (RTX 3050) |
|------|--------|---------|-----------------|
| **YOLOv11n** | **2.6M** | **0.971** | **35.7** |
| YOLOv11s | 9.4M | 0.975 | ~22 |
| YOLOv11m | 20.1M | 0.978 | ~11 |

### 各类别检测精度

| 类别 | mAP@0.5 | 类别 | mAP@0.5 |
|------|---------|------|---------|
| ripe_apple | 0.98 | ripe_grape | 0.98 |
| unripe_apple | 0.93 | unripe_grape | 0.95 |
| ripe_banana | 0.98 | ripe_orange | 0.97 |
| unripe_banana | 0.95 | unripe_orange | 0.94 |

### 品质分级规则

检测完成后，16 类细粒度标签自动映射为 ABC 三级：

| 等级 | 判定条件 | 建议处理 |
|------|---------|---------|
| 🟢 **A 级** | 成熟 + 置信度 ≥ 0.70 | 直接上市 / 精品礼盒装 |
| 🟡 **B 级** | 未成熟 / 置信度中等 / 低置信腐烂 | 超市散装 / 批发渠道 |
| 🔴 **C 级** | 腐烂（高置信度）/ 置信度 < 0.30 | 降级加工 / 废弃 |

---

## 功能模块

系统通过 Streamlit 提供 9 个功能 Tab，按角色动态显示：

| 模块 | 功能 |
|------|------|
| **📸 入库检测** | 上传水果图像，YOLOv11n 实时检测 + ABC 分级 |
| **📦 批次管理** | 创建入库批次、关联供应商、申报数量、累积检测 |
| **🔍 复核中心** | 经理查看待复核批次，逐批通过/拒收并留痕 |
| **📊 数据统计** | 入库量趋势、品质分布、供应商对账、CSV 导出 |
| **🏪 供应商管理** | 供应商档案 CRUD（名称、地域、联系方式） |
| **🍎 品类管理** | 水果品类维护（苹果/香蕉/葡萄/橙子） |
| **👤 用户管理** | 管理员创建/启停账号，分配角色 |
| **📋 操作日志** | 全量操作审计，只读不可改 |
| **📄 PDF 报表** | 基于 ReportLab 生成含图表的中文 PDF 检测报告 |

---

## 快速开始

### 环境要求

- Python 3.10
- Windows / macOS / Linux
- （可选）NVIDIA GPU + CUDA 11.8（CPU 推理亦可）

### 安装

```bash
# 克隆仓库
git clone https://github.com/mpc1589644842/kunkun.git
cd kunkun

# 安装依赖
pip install -r requirements.txt
```

### 运行

```bash
# 方式一：直接启动
python run_app.py

# 方式二：手动启动 Streamlit
streamlit run app.py --browser.gatherUsageStats false
```

浏览器访问 `http://localhost:8501`，使用演示账号登录：

| 用户名 | 密码 | 角色 |
|--------|------|------|
| `operator` | `op123` | 操作员 |
| `manager` | `mgr123` | 经理 |
| `admin` | `admin123` | 管理员 |

### 模型推理（编程调用）

```python
from ultralytics import YOLO

# 加载权重
model = YOLO("weights_backup/best_original.pt")

# 推理
results = model("your_image.jpg")
```

### ONNX Runtime 推理

```python
import onnxruntime as ort
import numpy as np
from PIL import Image

session = ort.InferenceSession("exported_models/model_fp32.onnx")
# ... 预处理与后处理见 app.py
```

---

## 数据集

采用**三阶段递进式**构建策略，总计 **9,737 张**标注图像：

| 阶段 | 数据来源 | 数量 | 特点 |
|------|---------|------|------|
| **Stage 1** | Kaggle 公开数据集 + AI 辅助标注 + 人工校验 | ~2,400 | 单目标、理想光照，冷启动 |
| **Stage 2** | 拼接合成多目标场景 | ~3,000 | 多目标密集场景，标签自动继承 |
| **Stage 3** | 专业标注团队真实场景采集 | ~4,337 | 复杂背景、多角度、真实仓储环境 |

- 标注格式：YOLO 格式（归一化边界框）
- 类别数：16 类（涵盖苹果/香蕉/葡萄/橙子的不同成熟度状态）
- 划分：训练集：验证集：测试集 = 8:1:1

---

## 项目结构

```
fruit_ripeness_std/
├── app.py                  # Streamlit 主入口（9 Tab 页面路由）
├── models.py               # 业务实体 dataclass 定义
├── grading.py              # ABC 三级品质分级 + 跨类 NMS
├── auth.py                 # 用户认证（bcrypt）
├── report_generator.py     # PDF 中文报告生成（ReportLab）
├── run_app.py              # 一键启动脚本
├── db_init.py              # 数据库初始化（建表 + 种子数据）
├── data.yaml               # 训练用数据集配置
│
├── db/                     # 数据持久化层
│   ├── orm_models.py       # SQLAlchemy ORM 模型
│   ├── session.py          # 数据库会话管理
│   └── repositories/       # Repository 模式实现
│       ├── batch_repo.py   #   批次仓库
│       ├── supplier_repo.py#   供应商仓库
│       └── user_repo.py    #   用户仓库
│
├── services/               # 业务逻辑层
│   ├── batch_service.py    #   批次状态机 + 三阈值判定
│   ├── permission_service.py # RBAC 权限控制
│   └── supplier_service.py #   供应商业务逻辑
│
├── pages/                  # UI 页面模块
│   ├── login_page.py       #   登录页
│   └── supplier_tab.py     #   供应商管理 Tab
│
├── scripts/                # 工具脚本集
│   ├── train.py            #   模型训练
│   ├── validate.py         #   模型验证
│   ├── export_quantize.py  #   ONNX 导出 + INT8 量化
│   ├── auto_label.py       #   AI 辅助自动标注
│   ├── mosaic_augment.py   #   拼接合成数据增强
│   ├── merge_datasets.py   #   多源数据集合并
│   ├── finetune.py         #   增量微调实验
│   └── ...                 #   其他数据处理脚本
│
├── weights_backup/         # 模型权重（PyTorch）
│   ├── best_original.pt    #   最佳模型（mAP@0.5=0.971）
│   ├── last_original.pt    #   最终 checkpoint
│   └── yolo11n.pt          #   YOLOv11n 预训练权重
│
├── exported_models/        # 模型导出（ONNX / TorchScript）
│   ├── model_fp32.onnx     #   FP32 ONNX
│   ├── model_int8.onnx     #   INT8 量化 ONNX
│   └── model.torchscript   #   TorchScript
│
└── paper_assets/           # 论文图表素材
```

---

## 单元测试

系统共编写 **149 个**单元测试，覆盖全部核心业务路径，持续保持 **100%** 通过率：

| 测试范围 | 用例数 |
|----------|--------|
| 批次仓库 (batch_repo) | 40+ |
| 供应商仓库 (supplier_repo) | 58+ |
| 用户仓库 (user_repo) | 39 |
| 状态机逻辑 | 12 |

---

## 实验记录

### 增量微调失败实验

曾尝试用 800 张富士苹果图片对基座模型做增量微调，以解决富士条纹被误判为腐烂的问题。结果出现**灾难性遗忘**（Catastrophic Forgetting）：

| 指标 | 微调前 | 微调后 | 变化 |
|------|--------|--------|------|
| 全类别 mAP@0.5 | **0.971** | 0.654 | **-0.317** |
| 富士苹果 mAP | 0.83 | 0.90 | +0.07 |

> **结论**：小数据集微调大范围破坏模型能力。最终选择保留基座模型，转而在业务流程层通过置信度阈值和人工复核兜底解决误判问题。

---

## 许可证

本项目采用 [MIT License](LICENSE)。

---

*2026 — 太原科技大学毕业设计*
