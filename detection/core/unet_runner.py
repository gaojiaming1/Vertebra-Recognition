# -*- coding: utf-8 -*-
"""
UNet 推理封装。
直接复用 UNet_Demo 项目里训练好的 unet.Unet 类（不重写网络结构），
通过把 UNet_Demo 根目录加入 sys.path 动态 import。
"""
import os
import sys
import numpy as np
from PIL import Image


# ------------------------------------------------------------
# UNet_Demo 源码根目录定位
#  - 开发模式用绝对路径（你本机的项目位置）
#  - 打包后（PyInstaller）用 _MEIPASS 里的副本，无需用户手动配置
# ------------------------------------------------------------
def _resolve_unet_demo_root():
    if getattr(sys, "frozen", False):
        bundled = os.path.join(sys._MEIPASS, "unet")
        if os.path.isdir(bundled):
            return bundled
    # 开发模式：本仓库 detection/unet 即为 UNet_Demo 源码目录
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "unet")


_UNET_DEMO_ROOT = _resolve_unet_demo_root()
# 必须「强制置顶」而不是「不在才插入」：
# 本项目自身也有个 utils/ 包，若 UNet_Demo 排在它后面，
# `from utils.utils import ...` 会被抢走 → ModuleNotFoundError: No module named 'utils.utils'
# 后续 YoloRunner._prepare_legacy_path 会把 yolov5_legacy 移到最前，两者互不干扰。
if _UNET_DEMO_ROOT in sys.path:
    sys.path.remove(_UNET_DEMO_ROOT)
sys.path.insert(0, _UNET_DEMO_ROOT)

from unet import Unet as _Unet  # 来自 UNet_Demo/unet.py


class UnetRunner:
    """
    UNet 推理封装：
    - 支持 4 个类别（背景 + chine + coccyx + cocc）
    - 默认 backbone = vgg
    - 默认输入尺寸 1024×1024
    - mix_type 决定输出图样式（0=blend 原图+mask，1=纯 mask，2=仅保留目标区域）
    """

    DEFAULT_NUM_CLASSES = 4
    DEFAULT_BACKBONE = "vgg"
    DEFAULT_INPUT_SHAPE = [1024, 1024]
    DEFAULT_CUDA = False  # 没 GPU 时改 False，避免 nn.DataParallel 报错

    def __init__(self, model_path, num_classes=None, input_shape=None,
                 backbone=None, cuda=None, mix_type=0):
        self.model_path = model_path
        self.num_classes = num_classes or self.DEFAULT_NUM_CLASSES
        self.input_shape = input_shape or self.DEFAULT_INPUT_SHAPE
        self.backbone = backbone or self.DEFAULT_BACKBONE
        # 自动判断 CUDA：有 GPU 且 cuda 未显式指定就用 cuda
        if cuda is None:
            try:
                import torch
                cuda = torch.cuda.is_available()
            except Exception:
                cuda = False
        self.cuda = cuda
        self.mix_type = mix_type  # 默认 blend（保留鱼骨+分割上色），与 test2 训练分布一致

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"UNet 权重文件不存在: {model_path}")

        # 实例化 UNet_Demo 的 Unet 类（它会在内部 load_state_dict）
        self.net = _Unet(
            model_path=model_path,
            num_classes=self.num_classes,
            backbone=self.backbone,
            input_shape=self.input_shape,
            cuda=self.cuda,
            mix_type=self.mix_type,
        )

    def predict(self, image):
        """
        对单张图片做推理。
        :param image: PIL.Image 或 numpy.ndarray (BGR)
        :return: PIL.Image (RGB)，根据 mix_type 不同样式不同
        """
        # 接受 numpy(BGR) 或 PIL，统一转 PIL
        if isinstance(image, np.ndarray):
            if image.ndim == 3 and image.shape[2] == 3:
                image = Image.fromarray(image[..., ::-1])  # BGR -> RGB
            else:
                image = Image.fromarray(image)

        if image.mode != "RGB":
            image = image.convert("RGB")

        result = self.net.detect_image(image)
        # 兼容返回值已经是 PIL 的情况
        if not isinstance(result, Image.Image):
            result = Image.fromarray(np.asarray(result))
        return result

    @staticmethod
    def get_default_unet_path():
        """给 GUI 默认权重路径用（兼容开发/打包两种模式）"""
        # 打包后用 _MEIPASS 里捆绑的权重
        if getattr(sys, "frozen", False):
            bundled = os.path.join(sys._MEIPASS, "models", "unet_best.pth")
            if os.path.exists(bundled):
                return bundled
        # 开发模式：权重放在 detection/models/ 下（需自行放置训练好的 best_epoch_weights.pth）
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base, "models", "unet_best.pth")