# -*- coding: utf-8 -*-
"""
YOLOv5 推理封装（兼容新老权重）：
- 优先使用 ultralytics 标准接口（适合 ultralytics>=8.x 训练的权重 / YOLOv8 / YOLO11）
- 如果加载失败（典型：老版 YOLOv5 权重，缺少 models.yolo 模块），自动切换到「老 YOLOv5」加载模式
  - 把 yugu-yolov5-master/yolov5-master/yolov5-master 加入 sys.path
  - 使用 DetectMultiBackend + NMS + Annotator
"""
import os
import sys
import numpy as np
from PIL import Image


# ------------------------------------------------------------
# 老 YOLOv5 项目根目录（用于老权重兜底加载）
#  - 开发模式用绝对路径
#  - 打包后（PyInstaller）用 _MEIPASS 里的副本
# ------------------------------------------------------------
def _resolve_yolov5_legacy_root():
    if getattr(sys, "frozen", False):
        bundled = os.path.join(sys._MEIPASS, "yolov5")
        if os.path.isdir(bundled):
            return bundled
    # 开发模式：本仓库 detection/yolov5 即为 vendored 的 YOLOv5 推理代码
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "yolov5")


_YOLOV5_LEGACY_ROOT = _resolve_yolov5_legacy_root()


# ------------------------------------------------------------
# 按类别分组的交替配色（BGR 顺序，给 cv2/Annotator 使用）
#   - 每个类别有一组专属颜色（类间颜色完全不同，一眼能区分类别）
#   - 同一类内循环交替使用（相邻同类目标颜色不同，方便逐个区分）
# ------------------------------------------------------------
_CLASS_PALETTE_BGR = {
    0: [(56, 56, 255), (0, 140, 255), (0, 255, 255)],    # chine：红 / 橙 / 黄
    1: [(0, 200, 0), (160, 255, 0), (200, 255, 120)],    # coccyx：绿系
    2: [(255, 0, 200), (200, 0, 255), (255, 120, 255)],  # cocc：紫品红系
}
_FALLBACK_PALETTE_BGR = [
    (56, 56, 255), (0, 200, 0), (255, 0, 200),
    (0, 140, 255), (160, 255, 0), (200, 0, 255),
]


# ------------------------------------------------------------
# 推理尺寸
# 实测（I3-1.jpg 1916x223，与 detect.py/detect2 对照）：
#     原图 imgsz=6400  → 90 / 39 / 36  ← 与 detect2 的 88/38/35 一致 ✅
#     原图 imgsz=3200  → 102 / 39 / 33（多检 chine、漏检 cocc）
#     原图 imgsz=1600  → 91 / 37 / 8   （cocc 几乎全漏）
#   所以保持 6400（这也是 detect.py 里跑出 88/38/35 时用的尺寸）。
# 可用环境变量 EEL_IMGSZ 覆盖，方便做对比实验。
# ------------------------------------------------------------
DEFAULT_IMGSZ = 6400

def _resolve_imgsz():
    import os
    v = os.environ.get("EEL_IMGSZ", "").strip()
    if v.isdigit():
        return int(v)
    return DEFAULT_IMGSZ


def _get_class_palette(c):
    """按类别返回该类的专属调色板（未知类别用兜底调色板）"""
    return _CLASS_PALETTE_BGR.get(int(c), _FALLBACK_PALETTE_BGR)


# ------------------------------------------------------------
#  绘制检测框（不再依赖 yolov5 的 Annotator，避免依赖 matplotlib/seaborn）
#  输入输出都用 RGB numpy，内部转 BGR 给 cv2（调色板也是 BGR 顺序）
# ------------------------------------------------------------
def _draw_boxes(rgb, detections, line_width=3, font_scale=0.7):
    try:
        import cv2
    except Exception:
        return rgb

    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    H, W = bgr.shape[:2]
    lw = max(1, int(round(line_width)))
    tf = max(lw - 1, 1)
    per_class_idx = {}

    # 按 x1 排序 → 空间相邻的同类目标在视觉顺序上也相邻，配色交替更直观
    for d in sorted(detections, key=lambda z: z.get("xyxy", [0])[0]):
        x1, y1, x2, y2 = [int(round(v)) for v in d.get("xyxy", [0, 0, 0, 0])]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)
        c = int(d.get("cls_id", -1))
        palette = _get_class_palette(c)
        i = per_class_idx.get(c, 0)
        color = palette[i % len(palette)]
        per_class_idx[c] = i + 1

        cv2.rectangle(bgr, (x1, y1), (x2, y2), color, lw, cv2.LINE_AA)

        label = d.get("label") or f"{d.get('class','')} {d.get('conf',0):.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                         fontScale=font_scale, thickness=tf)
        th = th + base + 3
        outside = y1 - th >= 3
        if outside:
            y_top, y_bot = y1 - th, y1
        else:
            y_top, y_bot = y1, y1 + th
        x_left = max(0, x1 - 1)
        x_right = min(x_left + tw + 6, W - 1)
        if x_right > x_left:
            cv2.rectangle(bgr, (x_left, y_top), (x_right, y_bot), color, -1, cv2.LINE_AA)
            cv2.putText(bgr, label, (x_left + 3, y_bot - base + (0 if outside else 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255),
                        tf, cv2.LINE_AA)

    return np.ascontiguousarray(bgr[:, :, ::-1])


class YoloRunner:
    """
    YOLOv5 推理封装：
    - 接受权重文件 (.pt)
    - 支持设置置信度阈值和 IoU 阈值
    - predict() 返回 (标注后的 PIL.Image, 检测结果列表)
    - 自动兼容老版 YOLOv5 权重（fallback 到 DetectMultiBackend）
    """

    def __init__(self, weights_path, conf_thres=0.25, iou_thres=0.45,
                 imgsz=None, device=""):
        """
        :param weights_path: YOLOv5 权重文件 (.pt)
        :param conf_thres: 置信度阈值
        :param iou_thres:   NMS IoU 阈值
        :param imgsz:       推理尺寸
        :param device:      cuda device (e.g. "0") / "cpu" / ""(自动)
        """
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"YOLOv5 权重文件不存在: {weights_path}")

        self.weights_path = weights_path
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = int(imgsz) if imgsz else _resolve_imgsz()
        self.device_arg = device

        # 先尝试新版 ultralytics；失败则切到老 YOLOv5 模式
        try:
            from ultralytics import YOLO  # noqa: F401  (仅探测可用性)
            self.model = YOLO(weights_path)
            self.class_names = getattr(self.model, "names", {})
            self.use_legacy = False
            print(f"[YoloRunner] 使用 ultralytics 标准接口加载: {weights_path}")
            print(f"[YoloRunner] 类别: {self.class_names}")
        except Exception as e:
            err_msg = str(e)
            # 老 YOLOv5 权重 + 新版 ultralytics 包的典型错误特征
            if ("NOT forwards compatible" in err_msg
                    or "No module named 'models'" in err_msg
                    or "No module named 'models.yolo" in err_msg
                    # 打包后：UNet_Demo 的 utils 抢占了同名包，导致 models.common 里
                    # `from utils import TryExcept` 失败（UNet 的 utils 里没有 TryExcept）
                    or "cannot import name 'TryExcept'" in err_msg
                    # 老 DetectionModel 没有 fuse(verbose=) 签名，说明是老 YOLOv5 结构
                    or "fuse() got an unexpected keyword" in err_msg
                    # 环境里根本没装 ultralytics（比如打包时没收进去）→ 直接走 legacy
                    or "No module named 'ultralytics'" in err_msg):
                print(f"[YoloRunner] ultralytics 加载失败（检测到老 YOLOv5 权重），"
                      f"切换到 legacy 模式…")
                self._init_legacy(weights_path, device)
            else:
                raise

    # ============================================================
    #  路径准备（在任何 torch.load 之前调用）
    # ============================================================
    @staticmethod
    def _prepare_legacy_path():
        """
        把老 yolov5 源码目录置顶，并清掉 utils/models 模块缓存。

        必须在 torch.load 之前调用！因为 pickle 反序列化老权重时会去找
        'models.yolo' 类 → import models.common → from utils import TryExcept。
        若此时 utils 仍缓存着 UNet_Demo 的版本（里面没有 TryExcept），就会 ImportError。
        """
        if _YOLOV5_LEGACY_ROOT in sys.path:
            sys.path.remove(_YOLOV5_LEGACY_ROOT)
        sys.path.insert(0, _YOLOV5_LEGACY_ROOT)

        for mod_name in list(sys.modules.keys()):
            if (mod_name == "utils" or mod_name.startswith("utils.")
                    or mod_name == "models" or mod_name.startswith("models.")):
                del sys.modules[mod_name]

    # ============================================================
    #  老 YOLOv5 fallback 初始化
    # ============================================================
    def _init_legacy(self, weights_path, device):
        """用老 YOLOv5 项目的代码加载权重"""
        # ---- 处理 sys.path 冲突 ----
        # UNet_Demo 也有个 utils/，会和老 YOLOv5 的 utils 冲突。
        # 解决：把老 yolov5 项目放到 sys.path 第一位，并清掉已缓存的 utils 包。
        self._prepare_legacy_path()

        import torch
        from models.common import DetectMultiBackend
        from utils.torch_utils import select_device

        if device:
            sel_device = torch.device(device)
        else:
            sel_device = select_device("")

        self.model = DetectMultiBackend(weights_path, device=sel_device)
        self.class_names = self.model.names
        self.use_legacy = True
        self._device = sel_device
        self._stride = self.model.stride
        self._pt = self.model.pt  # 是否是纯 PyTorch 模型
        print(f"[YoloRunner] legacy 模式加载成功: {weights_path}")
        print(f"[YoloRunner] 类别: {self.class_names}")

    # ============================================================
    #  单图推理（统一入口）
    # ============================================================
    def predict(self, image, draw_on=None):
        """
        :param image:   送进 YOLO 做检测的图（PIL.Image / RGB ndarray）
        :param draw_on: 把检测框画到哪张图上（PIL.Image）。None = 画在 image 自己身上。
                        典型用法：YOLO 对**原图**检测（yolo 就是在原图上训练的），
                        然后把框画到 UNet 语义分割效果图上。
        :return: (annotated_pil_image, detections_list)
                 detections_list: [{'class','cls_id','conf','xyxy','label'}, ...]

        带显存保护：遇到 CUDA OOM 会自动降分辨率重试，再不行就退回 CPU，
        保证「批量跑到第 N 张突然显存不足」不会让整批失败。
        """
        try:
            return self._predict_once(image, self.imgsz, self.device_arg, draw_on)
        except RuntimeError as e:
            msg = str(e)
            oom = ("out of memory" in msg.lower()) or ("not enough memory" in msg.lower())
            if not oom:
                raise
            self._free_cuda()
            # 逐级降级：imgsz/2 → imgsz/4 → CPU
            for scale in (0.5, 0.25):
                try:
                    imgsz = max(640, int(self.imgsz * scale))
                    print(f"[YoloRunner] 显存不足，自动降分辨率重试 imgsz={imgsz}")
                    return self._predict_once(image, imgsz, self.device_arg, draw_on)
                except RuntimeError as e2:
                    if "out of memory" not in str(e2).lower():
                        raise
                    self._free_cuda()
            print("[YoloRunner] 显存仍不足，退回 CPU 重试（速度会变慢）")
            self._free_cuda()
            return self._predict_once(image, max(640, int(self.imgsz * 0.25)), "cpu", draw_on)

    @staticmethod
    def _free_cuda():
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    @staticmethod
    def _as_rgb(image):
        """PIL / ndarray → RGB uint8 contiguous ndarray"""
        if isinstance(image, Image.Image):
            arr = np.array(image.convert("RGB"))
        else:
            arr = np.asarray(image)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.shape[2] == 4:
                arr = arr[..., :3]
        return np.ascontiguousarray(arr)

    def _predict_once(self, image, imgsz, device, draw_on=None):
        if self.use_legacy:
            return self._predict_legacy(image, imgsz=imgsz, device=device, draw_on=draw_on)
        return self._predict_new(image, imgsz=imgsz, device=device, draw_on=draw_on)

    def _finish(self, source_rgb, draw_rgb, detections):
        """
        统一收尾：把 detections 的坐标映射到 draw_rgb 的坐标系并画框。
        （YOLO 在原图上推理，draw 图若尺寸不同，按比例缩放坐标）
        """
        sh, sw = draw_rgb.shape[:2]
        oh, ow = source_rgb.shape[:2]
        if (sh, sw) != (oh, ow) and oh and ow:
            sx, sy = sw / ow, sh / oh
            for d in detections:
                x1, y1, x2, y2 = d["xyxy"]
                d["xyxy"] = [x1 * sx, y1 * sy, x2 * sx, y2 * sy]
        out = _draw_boxes(draw_rgb, detections)
        return Image.fromarray(out), detections

    # ----- 新版 ultralytics 推理 -----
    def _predict_new(self, image, imgsz=None, device=None, draw_on=None):
        imgsz = imgsz or self.imgsz
        device = device if device is not None else self.device_arg
        src_rgb = self._as_rgb(image)
        draw_rgb = self._as_rgb(draw_on) if draw_on is not None else src_rgb

        results = self.model.predict(
            source=src_rgb,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=imgsz,
            device=device if device else None,
            verbose=False,
        )

        result = results[0]
        detections = []
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i].item())
                conf = float(boxes.conf[i].item())
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                name = (self.class_names.get(cls_id, str(cls_id))
                        if isinstance(self.class_names, dict)
                        else (self.class_names[cls_id] if cls_id < len(self.class_names)
                              else str(cls_id)))
                detections.append({
                    "class": name,
                    "cls_id": cls_id,
                    "conf": conf,
                    "xyxy": [float(v) for v in xyxy],
                    "label": f"{name} {conf:.2f}",
                })
        return self._finish(src_rgb, draw_rgb, detections)

    # ----- 老 YOLOv5 推理（DetectMultiBackend + NMS，与 detect.py 流程严格对齐） -----
    def _predict_legacy(self, image, imgsz=None, device=None, draw_on=None):
        import torch
        from utils.general import non_max_suppression, scale_boxes
        from utils.augmentations import letterbox

        imgsz = imgsz or self.imgsz
        # 显存兜底：允许临时把模型搬到 CPU
        if device == "cpu":
            try:
                self.model.model.to("cpu")
                self._device = torch.device("cpu")
            except Exception:
                pass
        dev = self._device

        # 1) 统一转 RGB numpy
        src_rgb = self._as_rgb(image)
        draw_rgb = self._as_rgb(draw_on) if draw_on is not None else src_rgb

        # 2) 转 BGR —— 与 detect.py 完全一致：
        #    detect.py 用 cv2.imread 读进来就是 BGR，letterbox 在 BGR 上做。
        #    之前这里直接对 RGB 做 ::-1，等于把通道反着喂给模型 → 检测精度明显下降。
        im0_bgr = np.ascontiguousarray(src_rgb[:, :, ::-1])

        # 3) letterbox resize 到 imgsz（保比例 + 灰条，stride 对齐）
        stride = int(self._stride)
        imgsz = max(stride, (int(imgsz) // stride) * stride)
        img = letterbox(im0_bgr, imgsz, stride=stride, auto=True)[0]   # HWC, BGR

        # 4) BGR -> RGB, HWC -> CHW（detect.py 里是 transpose((2,0,1))[::-1]）
        img_chw = np.ascontiguousarray(img.transpose(2, 0, 1)[::-1])

        # 5) 转 tensor + 归一化
        img_t = torch.from_numpy(img_chw).to(dev)
        img_t = img_t.float() / 255.0
        if img_t.ndim == 3:
            img_t = img_t[None]

        # 6) 推理
        with torch.no_grad():
            pred = self.model(img_t, augment=False, visualize=False)

        # 7) NMS（与 detect.py 默认参数一致：agnostic=False, max_det=1000）
        pred = non_max_suppression(
            pred, self.conf_thres, self.iou_thres,
            classes=None, agnostic=False, max_det=1000,
        )[0]

        # 8) 解析
        detections = []
        if len(pred) and pred.numel():
            # 把 box 从 letterbox 尺寸缩回原图尺寸
            # scale_boxes 期望 (h, w)；img_chw 是 CHW，shape[1:] = (h, w)
            pred[:, :4] = scale_boxes(img_chw.shape[1:], pred[:, :4], src_rgb.shape).round()
            for det in pred.tolist():
                x1, y1, x2, y2, conf, cls = det
                c = int(cls)
                name = (self.class_names.get(c, str(c))
                        if isinstance(self.class_names, dict)
                        else (self.class_names[c] if c < len(self.class_names) else str(c)))
                detections.append({
                    "class": name,
                    "cls_id": c,
                    "conf": float(conf),
                    "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                    "label": f"{name} {conf:.2f}",
                })

        return self._finish(src_rgb, draw_rgb, detections)

    # ============================================================
    #  静态工具
    # ============================================================
    @staticmethod
    def get_default_yolo_path():
        """给 GUI 默认权重路径用（兼容开发/打包两种模式）"""
        # 打包后用 _MEIPASS 里捆绑的权重文件
        if getattr(sys, "frozen", False):
            bundled = os.path.join(sys._MEIPASS, "models", "yolo_best.pt")
            if os.path.exists(bundled):
                return bundled
        # 开发模式：权重放在 detection/models/ 下（需自行放置训练好的 best.pt）
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, "models", "yolo_best.pt")