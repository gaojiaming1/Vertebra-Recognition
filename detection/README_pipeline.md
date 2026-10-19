# 黄鳝椎骨智能识别系统（UNet → YOLOv5 串联推理）

把训练好的两个模型串联起来：一个 PyQt5 桌面应用（带开屏动画 + 全新美化界面）。

**流水线**：原图 → UNet 语义分割（Blend 混合效果图，保留整条鱼骨 + 分割上色）→
**YOLOv5 对原图做检测**（与 detect.py/detect2 一致，因为权重就是在原图上训练的）→
**检测框画在 UNet 分割效果图上**（一张图同时看到分割与检测，同类相邻目标交替配色）。

## 界面说明（v2.0）

- **开屏页**：脊椎骨动画 + 模型加载进度，模型加载完成后自动淡入主界面
- **权重已内置**：界面不再有 UNet/YOLO 权重选择，只有 **置信度阈值** 和 **IoU 阈值** 两个滑块
- **输出目录**：可自由选择（对话框里可新建文件夹），启动时会做「可写性预检」；
  打包成 exe 后默认输出到 `文档\黄鳝椎骨识别结果`（exe 的程序目录是临时目录，不能存结果）
- **批量容错**：单张失败只跳过该张并记录原因，不会中断整批；失败原因会写进日志
- **结果统计**：自动生成 `检测统计.csv`（每行一条鱼的各部位数量）；每条鱼的数量同时实时打印在界面日志里

---

## 项目结构

```
D:\prcharm project\unet_yolo_pipeline\
├── main.py                  # 启动入口：开屏页 → 主界面（PyCharm 里跑这个）
├── gui/
│   ├── splash.py            # 开屏页（脊椎动画 + 加载进度）
│   ├── theme.py             # 统一视觉主题（QSS）
│   ├── main_window.py       # PyQt5 主窗口
│   └── __init__.py
├── core/
│   ├── unet_runner.py       # UNet 推理封装（复用 UNet_Demo）
│   ├── yolo_runner.py       # YOLOv5 推理封装（legacy 兜底 + 显存保护）
│   ├── pipeline.py          # 串联流水线（单图/批量/统计/输出目录预检）
│   └── __init__.py
├── output/                  # 开发模式默认输出目录
├── build_exe.bat            # 一键打包
└── README.md
```

---

## 在 PyCharm 里跑

1. **打开 PyCharm**，File → Open → 选 `D:\prcharm project\unet_yolo_pipeline`
2. **配 Python 解释器**：建议用你 UNet_Demo 已经用过的那个 venv（`UNet_Demo/UNet_Demo/venv`）
   - 因为它已经装了 torch 等依赖；额外补 `pip install ultralytics PyQt5`
3. **右键 main.py → Run** 即可启动 GUI

或者命令行：
```bash
cd "D:\prcharm project\unet_yolo_pipeline"
python main.py
```

---

## 使用步骤

1. **选输入**：单张图片 或 整个文件夹（批量支持勾选「包含子文件夹」，自动跳过软件自己产出的结果图）
2. **选输出目录**：结果图（`原图 / UNet分割 / YOLO检测` 三张）+ 检测统计.csv保存到这里
3. **调阈值**：置信度（默认 0.25）、IoU（默认 0.45），运行中改动下次运行即生效
4. **点「▶ 开始处理」**：三栏实时显示结果，底部看进度、统计和日志
5. 点「打开目录」可直接跳到输出文件夹；双击任意结果图可放大查看

---

## ⚠ 重要：测试图必须与 YOLO 训练分布一致

实测发现：你的 **YOLO 训练分布** ≠ UNet 测试集分布，所以乱拿图测试会「检测不到」。

| 数据集 | 尺寸 | 用途 |
|---|---|---|
| `quanbu/images/train/*.jpg` | **1515×320 长条形 RGB** | YOLO 训练图 |
| `detect/Test*.bmp` | **1620×941 大 RGB** | YOLO 测试图（detect2.py 默认 source） |
| `Medical_Datasets/Images/*.png` | **512×512 灰度** | UNet 测试集 |

**实测结果**（同一张 `detect/Test1-1.bmp`，1620×941 RGB）：

| 流水线 | 检测数 |
|---|---|
| 直接 YOLO（原图） | **72** 个 |
| UNet mask → YOLO | **70** 个（几乎一致）|
| 拿错图（512×512 灰度） → YOLO | **0** 个（**不是软件 bug**）|

**所以**：在 GUI 里测试时，请选 `detect/` 目录（1020×94 这种大 RGB 图）—— 这是 YOLO 真正训练分布。不要拿 `Medical_Datasets/Images/`（512×512 灰度 X 光）来测，那是 UNet 的图。

---

## 模型配置

`core/unet_runner.py` 顶部有默认值，对应你的 UNet_Demo 训练设置：

| 参数 | 默认值 | 说明 |
|---|---|---|
| `num_classes` | 4 | 背景 + chine + coccyx + cocc |
| `backbone` | `vgg` | UNet 主干 |
| `input_shape` | `[1024, 1024]` | UNet 输入尺寸 |
| `mix_type` | 1 | 0=blend，1=纯 mask，2=抠图 |
| `cuda` | 自动检测 | 没 GPU 自动用 CPU |

如果你的训练设置不是 vgg，修改 `core/unet_runner.py` 里 `DEFAULT_BACKBONE = "resnet50"`（或对应 backbone）。

---

## 核心原理

```python
# 简化版流水线（实际代码见 core/pipeline.py）
from PIL import Image
from core import UnetRunner, YoloRunner

unet = UnetRunner(model_path="best_epoch_weights.pth")
yolo = YoloRunner(weights_path="best.pt")

img = Image.open("test.jpg").convert("RGB")
mask = unet.predict(img)            # ← UNet 输出
annotated, dets = yolo.predict(mask) # ← mask 作为 YOLO 输入
print(dets)  # [{'class': 'cocc', 'conf': 0.87, 'xyxy': [..]}, ...]
```

---

## 常见问题

**Q: 启动报错 `No module named 'PyQt5'`**
A: `pip install PyQt5`（如果用 UNet_Demo 自带 venv，记得在那个环境里装）

**Q: UNet 加载报 shape 不匹配**
A: 训练时 num_classes 和 backbone 必须与权重匹配。打开 `core/unet_runner.py` 修改 `DEFAULT_NUM_CLASSES` 和 `DEFAULT_BACKBONE`

**Q: YOLO 检测空（detections 为空）**
A: 把置信度调低（比如 0.1）；或者 mask 输出太干净导致 YOLO 没目标，可以换 mix_type=0（blend）重试

**Q: 批量处理时部分图片失败**
A: 现在失败原因会直接显示在日志里（每张单独跳过，不影响其他图）。常见原因：
   - 图片损坏 / 截断（`Truncated File Read`）
   - 特殊格式（16bit TIFF、CMYK）——已做兼容转换，若仍失败请发日志
   - 显存不足——软件会自动降分辨率重试，再不行退回 CPU

**Q: 输出目录写不进去**
A: 界面上「输出目录」下方会实时显示 ✅可写 / ❌原因。打包版默认写到 `文档\黄鳝椎骨识别结果`，别选系统保护目录（如 C:\ 根目录、Program Files）。

**Q: 想换测试图片**
A: 单张模式直接换；批量模式把所有图片放一个文件夹即可

---

## 下一步可扩展

- [ ] 视频/摄像头实时检测（cv2.VideoCapture）
- [ ] 导出 ONNX 加速推理
- [x] 检测结果 CSV 导出（已实现：输出目录里自动生成 `检测统计.csv`）
- [ ] 摄像头检测叠加 UNet mask 半透明显示
- [x] 打包成 .exe（见下方「打包成 exe」章节）

---

## 打包成 exe（发给没装 Python 的人用）

**一键打包**：双击项目根目录的 `build_exe.bat`（自动用 Yolo5 环境打包），或在命令行执行：

```bash
cd "D:\prcharm project\unet_yolo_pipeline"
D:\anaconda\envs\Yolo5\python.exe -m PyInstaller --clean -y unet_yolo_pipeline.spec
```

**产物**：`dist\unet_yolo_pipeline.exe`（单文件，含 torch + PyQt5 + 两个模型的源码和权重，
当前约 2.0 GB）

exe 自带**专属图标**：深潭青渐变底 + 鱼类脊椎 + 金色检测框（与开屏页同款），
资源管理器、任务栏、Alt-Tab 都会显示它；右键 exe → 属性里能看到产品名与 V2.0 版本号。
图标由 `tools/make_icon.py` 程序化生成（想换样式改配色重跑即可）：

```bash
python tools/make_icon.py      # 生成 assets/app.ico（含 16~256 共 9 个尺寸）
```

**发给别人**：直接把这个 exe 发出去即可，对方**不需要装 Python / torch / 任何环境**，双击就能用：
- exe 内置了你训练好的 UNet 权重（`best_epoch_weights.pth`）和 YOLO 权重（`best.pt`），启动后自动填好路径
- 检测框配色已内置（同类相邻目标交替变色），无需额外配置

**说明**：
1. 打包配置在 `unet_yolo_pipeline.spec`：只打包了 UNet_Demo/yolov5 的**推理必需源码 + 权重**（venv、数据集、训练日志都没打包，避免 exe 膨胀到十几 GB）
2. 第一次双击 exe 启动较慢（单文件模式要把运行环境解压到临时目录，约 30~60 秒），之后正常
3. 如果换权重：exe 里默认内置 `exp45/best.pt`；GUI 里手动「浏览」选新的 `.pth` / `.pt` 文件即可，无需重新打包
4. 打包脚本 `build_exe.bat` 会自动：检测并安装 PyInstaller → 清理旧构建 → 打包 → 提示产物位置
5. 图标与版本信息由 spec 里的 `icon=assets/app.ico` 和 `version=version_info.txt` 指定，
   改 exe 属性里的产品名等信息直接编辑 `version_info.txt` 后重新打包
---

## 安装程序（推荐给别人的分发方式）

`dist\安装包\` 文件夹里有两个文件，**一起拷给别人**（放在同一个文件夹）：

| 文件 | 大小 | 作用 |
|---|---|---|
| `鱼类脊椎骨识别_V2.0_安装程序.exe` | 35 MB | 双击运行，就是安装向导 |
| `unet_yolo_pipeline.exe` | 约 2.6 GB | 主程序本体（安装源，不要单独删） |

### ⚠ 不要为了瘦身删 torch 的 CUDA 库

曾为了把 exe 从 2.9G 压到 2.0G，排除了 `cusolver / cufft / cusparse / cudnn 训练库 / nvfuser`
等 CUDA DLL。本机测试完全正常，但换到**没装 CUDA Toolkit 的电脑**直接崩：

```
OSError: [WinError 126] 找不到指定的模块。
Error loading "...\_MEIxxx\torch\lib\shm.dll" or one of its dependencies.
```

`shm.dll / c10_cuda.dll` 的依赖链里包含这些库；开发机装了 CUDA Toolkit，系统 PATH 里
本来就有它们，所以"本机测试通过"是假象。现在 spec 里 `_DROP_DLL_PATTERNS = []`，
**一个 DLL 都不排除**。以后若要瘦身，必须先在没装 CUDA 的机器上验证。

每次打包后建议跑依赖完整性检查（静态解析 PE 导入表，不用换电脑试）：

```bash
python tools/check_deps.py build_v6/unet_yolo_pipeline
```

双击安装程序后的向导流程：**欢迎 → 选择安装位置 → 附加任务 → 安装 → 完成**

- 可自己改安装目录（默认 `C:\Users\<用户名>\AppData\Local\Programs\鱼类脊椎骨识别`，不需要管理员权限）
- 可勾选「桌面快捷方式」「开始菜单快捷方式」「安装后运行」
- 安装完会在 Windows「设置 → 应用」里生成卸载项，也可以运行安装目录里的 `卸载.bat`
- 快捷方式通过 COM 接口创建，不依赖脚本宿主，公司电脑有限制也能装

命令行（静默安装，批量部署用）：

```bat
鱼类脊椎骨识别_V2.0_安装程序.exe --dir="D:\Software\鱼类脊椎骨识别" --no-run
:: 可选： --no-desktop  --no-startmenu  --no-run
:: 安装日志写在目标目录的 install_log.txt
```

> 为什么不是「一个 exe」？
> 主程序是 PyInstaller 单文件 exe，它的启动引导程序要在**文件末尾**找自己的归档标记，
> 无法像 7z 自解压那样把主程序追加到安装程序尾部（追加后安装程序自己会启动不了）。
> 所以采用「安装程序 + 同目录主程序」两个文件的形式（很多大型软件也是这样）。
> 若以后能联网装 Inno Setup / 7-Zip，可以再合成真正的单文件安装包。

### 桌面快捷方式（不想走安装向导的话）

手动：右键 `unet_yolo_pipeline.exe` →「发送到」→「桌面快捷方式」。
换图标：右键快捷方式 → 属性 → 更改图标 → 选 `assets\app.ico`。

### 图标没更新？

Windows 会缓存 exe 图标。双击运行 `tools\刷新图标缓存.bat`（会重启资源管理器），
或把 exe 复制到一个新文件夹再看（换路径会强制重读图标）。

---

## 安装程序（推荐的分发方式）

`dist\安装包\` 文件夹里有两个文件，**一起拷给别人**（放在同一个文件夹）：

| 文件 | 大小 | 作用 |
|---|---|---|
| `鱼类脊椎骨识别_V2.0_安装程序.exe` | 35 MB | 双击运行，就是安装向导 |
| `unet_yolo_pipeline.exe` | 约 1.9 GB | 主程序本体（安装源，不要单独删） |

双击安装程序后的流程：**欢迎 → 选择安装位置 → 附加任务 → 安装 → 完成**

- 可自己改安装目录（默认 `C:\Users\<用户名>\AppData\Local\Programs\鱼类脊椎骨识别`，不需管理员权限）
- 可勾选「桌面快捷方式」「开始菜单快捷方式」「安装后运行」
- 安装完会在 Windows「设置 → 应用」里生成卸载项，也可运行安装目录里的 `卸载.bat`
- 快捷方式通过 COM 接口创建，不依赖脚本宿主，有脚本限制的电脑也能装

命令行（静默安装，批量部署用）：

```bat
鱼类脊椎骨识别_V2.0_安装程序.exe --dir="D:\Software\鱼类脊椎骨识别" --no-run
:: 可选： --no-desktop  --no-startmenu  --no-run
:: 安装日志写在目标目录的 install_log.txt
```

> 为什么不是「一个 exe」？
> 主程序是 PyInstaller 单文件 exe，它的启动引导程序要在**文件末尾**找自己的归档标记，
> 无法像 7z 自解压那样把主程序追加到安装程序尾部（追加后安装程序自己会启动不了）。
> 所以采用「安装程序 + 同目录主程序」两个文件的形式（很多大型软件也是这样）。
> 以后若能联网装 Inno Setup / 7-Zip，可再合成真正的单文件安装包。

### 桌面快捷方式（不想走安装向导的话）

手动：右键 `unet_yolo_pipeline.exe` →「发送到」→「桌面快捷方式」。
换图标：右键快捷方式 → 属性 → 更改图标 → 选 `assets\app.ico`。

### 图标没更新？

Windows 会缓存 exe 图标。双击运行 `tools\刷新图标缓存.bat`（会重启资源管理器），
或把 exe 复制到一个新文件夹再看（换路径会强制重读图标）。

---

## 出问题怎么排查（重要）

exe 是 GUI 程序，**看不到控制台**，所以做了两件事：

1. **运行日志**：所有输出都会同时写进
   `C:/Users/<用户名>/AppData/Local/EelVertebra/app.log`
2. **崩溃提示**：任何未捕获异常都会弹窗，告诉你错误摘要和日志位置，不再「静默闪退」

如果双击 exe 后**直接闪退、什么都没有**，按以下顺序排查：

| 现象 | 可能原因 | 处理 |
|---|---|---|
| 双击后无任何反应 | 单文件 exe 首次要解压约 2GB 到临时目录，慢（30~60 秒） | 等 1 分钟再看任务栏（图标已换成专属图标） |
| 解压到一半没了 | C 盘剩余空间不足（需要 3GB 以上空闲） | 清理临时目录 `%TEMP%` |
| 进程被杀 | Windows Defender / 360 等杀软拦截未签名的大 exe | 把 exe 加入白名单/信任区 |
| 闪退后无提示 | 老版本没有日志；用新版本再跑一次 | 查看 `app.log` 最后几行 |
| 显存不足报错 | 一次开多个实例，或有别的程序占着显卡 | 关掉其他占用 GPU 的程序后重跑 |
