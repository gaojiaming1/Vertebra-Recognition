# detection/models/ — 模型权重放置位置

本目录用于存放**训练好的模型权重**。由于 GitHub 单文件 100 MB 限制，权重**不纳入仓库**，
请自行训练后把以下两个文件放到这里：

| 文件 | 用途 | 对应模型 |
|---|---|---|
| `unet_best.pth` | UNet 语义分割权重 | UNet（backbone: vgg，4 类：背景/chine/coccyx/cocc） |
| `yolo_best.pt` | YOLOv5 检测权重 | YOLOv5（在长条 RGB 原图上训练） |

## 路径对应关系
默认路径在代码中写死为相对路径，clone 后无需改动：
- `detection/core/unet_runner.py` → `get_default_unet_path()` 返回 `detection/models/unet_best.pth`
- `detection/core/yolo_runner.py` → `get_default_yolo_path()` 返回 `detection/models/yolo_best.pt`

若你的权重文件名/位置不同，改这两个函数的返回值，或在 GUI 中手动「浏览」选择即可。

## 训练权重来源
- UNet：用 `detection/unet/train.py`（或 `train_medical.py`）在标注数据上训练，
  最佳权重存为 `logs/best_epoch_weights.pth`，重命名为 `unet_best.pth` 放在本目录。
- YOLOv5：用 `detection/yolov5/train.py` 训练，取 `runs/train/exp*/weights/best.pt`，
  重命名为 `yolo_best.pt` 放在本目录。

> 注意：测试图片需与 YOLO **训练分布一致**（长条 ~1515×320 RGB 图）。
> 用错分布（如 512×512 灰度 X 光）会检测不到目标——这不是软件 bug。
