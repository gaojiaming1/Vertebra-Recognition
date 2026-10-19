# AlloVert · 黄鳝椎骨自动计数与生长预测（论文配套代码）

> 中文说明见下方「中文」。English below.

This repository contains the **runnable code** accompanying the paper
**“Automated Vertebral Counting and Growth Prediction in Swamp Eel
(*Monopterus albus*)”**. It is organized for clarity and direct execution:
clone → install → (drop in your trained weights) → run.

The repository has three self-contained parts:

| Folder | What it is | How to run |
|---|---|---|
| `detection/` | **UNet → YOLOv5** vertebra-counting pipeline (PyQt5 desktop GUI) — the most complete, ready-to-run module | `python detection/main.py` |
| `growth_prediction/allovert/` | **AlloVert** ensemble (GRU + XGBoost + Stacking/Bagging) for bidirectional length ↔ weight prediction | `python RNN-yugu.py` etc. |
| `growth_prediction/s3cmr/` | **S3-CMR** (Sum-Constrained Multi-target Regression) reverse inference from morphology → vertebra count | `python S3-CMR.py` |
| `figures/` | Paper figures (Fig 2–10, Supplementary S1–S6) as SVG/PDF | — |

---

## 1. Vertebra detection (`detection/`)

A PyQt5 application that chains two trained models:

```
original image → UNet semantic segmentation → YOLOv5 detection on the segmentation map
                → annotated result (segmentation + detection overlaid)
```

### Install
```bash
pip install -r detection/requirements.txt
```
(uses your existing env with torch; if using the original UNet_Demo venv, just
`pip install ultralytics PyQt5` there).

### Run
```bash
python detection/main.py
```
The GUI lets you pick an input image/folder, set confidence/IoU thresholds,
and writes `原图 / UNet分割 / YOLO检测` results plus a `检测统计.csv`.

### Model weights (required, **not** in this repo)
Place your trained weights here (GitHub’s 100 MB/file limit prevents committing them):
```
detection/models/unet_best.pth     # UNet segmentation weights
detection/models/yolo_best.pt      # YOLOv5 detection weights
```
See `detection/models/README.md` for details. If the paths differ, edit the
defaults in `detection/core/unet_runner.py` / `detection/core/yolo_runner.py`.

> ⚠️ Test images must match the **YOLO training distribution** (long ~1515×320
> RGB strip images), otherwise detection returns nothing. A small sample image
> is provided in `detection/sample_data/`.

---

## 2. Growth prediction (`growth_prediction/`)

### AlloVert (`allovert/`)
Models: `CNN-yugu.py`, `DNN-yugu.py`, `FT.py`, `RNN-yugu.py` (GRU),
`TLNN.py`, `XGBoost.py`, `SVR.py`, plus `Fig+AlloVert.py` (result plotting).
Each script reads `train.xlsx` / `test.xlsx` from its own folder.

```bash
cd growth_prediction/allovert
python RNN-yugu.py          # trains/evaluates the GRU model, saves best_*.pth
```
`Fig+AlloVert.py` additionally needs the 7 model-prediction CSVs
(`RNN_yugu_predictions.csv`, …) — run the model scripts first, or drop the CSVs
into `allovert/data/`. A small sample of `体重分组训练集.xlsx` is included there.

### S3-CMR (`s3cmr/`)
`S3-CMR.py` (main), `GP.py`, `GPR.py`, `MLP.py`, `RF.py`, `RF + Logit.py`,
`SVR.py`, `XGBoost.py`, `Fig.py`. Reads `体重分组训练集.xlsx` / `体重分组测试集.xlsx`
(already copied into this folder) and writes reports to `./output_*`.

```bash
cd growth_prediction/s3cmr
python S3-CMR.py
```

Install shared deps:
```bash
pip install -r requirements.txt
```

---

## 3. Figures
All paper figures are in `figures/` (main figures) and `figures/supplementary/`
(S1–S6). They are vector SVG + PDF and can be opened directly.

---

## Repository layout
```
AlloVert-YOLOv5/
├── README.md
├── requirements.txt                 # growth-prediction deps
├── .gitignore
├── detection/                       # ★ most complete: UNet→YOLOv5 GUI
│   ├── main.py
│   ├── README_pipeline.md           # original pipeline README
│   ├── requirements.txt             # detection deps
│   ├── gui/  core/  tools/  utils/  assets/
│   ├── unet/                       # vendored UNet_Demo source (inference)
│   ├── yolov5/                      # vendored YOLOv5 inference code
│   ├── models/                      # ← put unet_best.pth, yolo_best.pt here
│   └── sample_data/                 # one small test image
├── growth_prediction/
│   ├── allovert/  (AlloVert models + data/)
│   └── s3cmr/     (S3-CMR models)
└── figures/  +  figures/supplementary/
```

## License
Code is provided for research reproducibility. Pretrained weights and raw CT
data are not redistributed here; obtain them from the authors.

---
# 中文

本仓库收录论文《黄鳝（*Monopterus albus*）椎骨自动计数与生长预测》的**可运行代码**，
按"克隆 → 安装依赖 → 放入训练权重 → 直接运行"的思路整理，结构清晰、各模块独立。

三部分：
- **`detection/`** — UNet→YOLOv5 椎骨识别桌面程序（PyQt5 GUI），**最完整、可直接运行**的部分。
- **`growth_prediction/allovert/`** — AlloVert 集成模型（GRU + XGBoost + Stacking/Bagging），体长↔体重双向预测。
- **`growth_prediction/s3cmr/`** — S3-CMR（Sum-Constrained Multi-target Regression）由形态反推椎数。

### 快速开始
1. 检测：`pip install -r detection/requirements.txt` → 把权重放入 `detection/models/`（见该目录 README）→ `python detection/main.py`。
2. 生长预测：`pip install -r requirements.txt` → 进入 `allovert/` 或 `s3cmr/` 运行对应脚本。

### 关于权重
训练好的 `unet_best.pth` / `yolo_best.pt` **未包含在仓库内**（GitHub 单文件 100 MB 限制），
请自行训练后放入 `detection/models/`。生长预测模型读取各自训练脚本产出的 `best_*.pth`。
