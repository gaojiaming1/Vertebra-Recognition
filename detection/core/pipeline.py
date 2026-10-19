# -*- coding: utf-8 -*-
"""
串联流水线：单张图片走「原图 → UNet mask → YOLOv5 检测」。

设计要点（针对批量失败 / 输出目录失败做过全面加固）：
  1. 读图用 _load_image()，兼容 16bit / CMYK / P / L / 灰度 TIFF 等非常规模式，
     单张损坏不会拖垮整批。
  2. 每张图独立 try/except，失败会记录 错误原因 + traceback，并继续跑下一张。
  3. 输出目录在开始前做「可写性预检」，写不进去会立刻给出明确原因，
     而不是跑完 N 张才发现一张都没存下。
  4. 自动跳过本软件自己产出的 *_1_original / *_2_unet_mask / *_3_yolo 文件，
     避免输出目录放在输入目录内时反复自喂。
"""
import os
import time
import csv
import traceback

import numpy as np
from PIL import Image

from .unet_runner import UnetRunner
from .yolo_runner import YoloRunner


class Pipeline:
    """
    单/批流水线。
    用法：
        pipe = Pipeline(unet_path, yolo_path)
        r = pipe.run_one("img/test.jpg", save_dir="output")
        pipe.run_batch(["img/1.jpg"], save_dir="output", progress_cb=lambda i,n,p: ...)
    """

    SUPPORTED_EXTS = (
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp", ".jfif", ".jpeg2000",
    )
    # 本软件自己产出的后缀，扫描输入目录时跳过（防止输出目录嵌套在输入目录里造成自喂）
    OUTPUT_SUFFIXES = ("_1_original", "_2_unet_mask", "_3_yolo")

    def __init__(self, unet_path, yolo_path, **kwargs):
        self.unet = UnetRunner(model_path=unet_path, **kwargs.get("unet_kwargs", {}))
        self.yolo = YoloRunner(weights_path=yolo_path, **kwargs.get("yolo_kwargs", {}))
        # YOLO 检测的输入源：
        #   "original" = 对原图检测（推荐！你的 yolo 就是在原图上训练的）
        #   "unet"     = 对 UNet 分割效果图检测（旧行为）
        # 无论哪种，检测框最终都画在 UNet 语义分割效果图上。
        self.detect_on = "original"

    # ------------------------------------------------------------
    #  参数热更新（不重新加载模型）
    # ------------------------------------------------------------
    def set_params(self, conf_thres=None, iou_thres=None, mix_type=None,
                   detect_on=None, imgsz=None):
        if conf_thres is not None:
            self.yolo.conf_thres = float(conf_thres)
        if imgsz is not None:
            self.yolo.imgsz = int(imgsz)
        if detect_on in ("original", "unet"):
            self.detect_on = detect_on
        if iou_thres is not None:
            self.yolo.iou_thres = float(iou_thres)
        if mix_type is not None:
            self.unet.mix_type = int(mix_type)
            # UNet_Demo 的 Unet 对象内部也存了一份 mix_type，必须同步
            try:
                self.unet.net.mix_type = int(mix_type)
            except Exception:
                pass

    # ------------------------------------------------------------
    #  读图（兼容各种非常规图像模式）
    # ------------------------------------------------------------
    @staticmethod
    def _to_rgb(im):
        """把任意模式的 PIL 图安全转成 RGB（16bit / CMYK / P / L / F 都能处理）"""
        if im.mode == "RGB":
            return im
        try:
            return im.convert("RGB")
        except Exception:
            pass
        # 兜底：走 numpy
        arr = np.asarray(im)
        if arr.ndim == 2:                       # 灰度
            arr = np.stack([arr] * 3, axis=-1)
        else:
            arr = arr[..., :3]
        if arr.dtype != np.uint8:               # 16bit / float
            arr = arr.astype(np.float32)
            mn, mx = float(arr.min()), float(arr.max())
            if mx <= mn:
                arr = np.zeros(arr.shape, dtype=np.uint8)
            else:
                arr = ((arr - mn) / (mx - mn) * 255.0).clip(0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    @classmethod
    def _load_image(cls, path):
        """安全读取一张图，返回 PIL RGB。失败会抛出带文件名的异常"""
        try:
            im = Image.open(path)
            im.load()          # 立即解码，损坏文件在这一步才会报错
            return cls._to_rgb(im)
        except Exception as e:
            raise RuntimeError(f"无法读取图片（{type(e).__name__}: {e}）") from e

    # ------------------------------------------------------------
    #  单图
    # ------------------------------------------------------------
    def run_one(self, image_path, save_dir=None, verbose=True):
        """
        单图完整流程。
        :return: dict {
            'original', 'mask', 'annotated', 'detections', 'elapsed', 'source_name'
        }
        """
        t0 = time.time()

        original = self._load_image(image_path)
        if verbose:
            print(f"[Pipeline] {os.path.basename(image_path)}: UNet 语义分割...")
        mask = self.unet.predict(original)

        if verbose:
            src = "原图" if self.detect_on == "original" else "UNet 分割图"
            print(f"[Pipeline] {os.path.basename(image_path)}: YOLOv5 检测（输入={src}，"
                  f"imgsz={self.yolo.imgsz}，conf={self.yolo.conf_thres}，iou={self.yolo.iou_thres}）")

        # YOLO 在「它训练时见过的那种图」上做检测（默认原图），
        # 检测框一律画到 UNet 语义分割效果图上 → 一张图同时看到分割与检测。
        detect_input = mask if self.detect_on == "unet" else original
        annotated, detections = self.yolo.predict(detect_input, draw_on=mask)

        elapsed = time.time() - t0

        result = {
            "original": original,
            "mask": mask,
            "annotated": annotated,
            "detections": detections,
            "elapsed": elapsed,
            "source_name": os.path.basename(image_path),
            "source_path": image_path,
        }

        if save_dir:
            self._save(result, save_dir)
        return result

    # ------------------------------------------------------------
    #  批量
    # ------------------------------------------------------------
    def run_batch(self, image_paths, save_dir=None, progress_cb=None):
        """
        批量跑。任何一张失败都只影响自己，不会中断整批。
        :param progress_cb: callback(done, total, last_result)
        :return: list[result]，失败项为 {'source_name','error','traceback'}
        """
        if save_dir:
            ok, msg = ensure_output_dir(save_dir)
            if not ok:
                raise RuntimeError(f"输出目录不可用：{save_dir}\n原因：{msg}")

        results = []
        total = len(image_paths)
        for i, p in enumerate(image_paths, 1):
            try:
                r = self.run_one(p, save_dir=save_dir, verbose=False)
                results.append(r)
            except Exception as e:
                tb = traceback.format_exc()
                print(f"[Pipeline] ❌ {p} 处理失败: {e}\n{tb}")
                results.append({
                    "source_name": os.path.basename(p),
                    "source_path": p,
                    "error": str(e),
                    "traceback": tb,
                })
            if progress_cb:
                try:
                    progress_cb(i, total, results[-1])
                except Exception:
                    pass

        if save_dir:
            try:
                write_summary_csv(results, save_dir)
            except Exception as e:
                print(f"[Pipeline] 统计 CSV 写入失败（不影响结果图）: {e}")
        return results

    # ------------------------------------------------------------
    #  目录扫描
    # ------------------------------------------------------------
    @classmethod
    def list_images(cls, folder, recursive=False):
        """列出文件夹下所有支持的图片（跳过本软件自己的输出文件），自然排序"""
        if not folder or not os.path.isdir(folder):
            return []

        files = []
        if recursive:
            for root, dirs, fs in os.walk(folder):
                dirs[:] = [d for d in dirs if not d.startswith(".")]
                for f in fs:
                    if f.lower().endswith(cls.SUPPORTED_EXTS):
                        files.append(os.path.join(root, f))
        else:
            for f in os.listdir(folder):
                fp = os.path.join(folder, f)
                if os.path.isfile(fp) and f.lower().endswith(cls.SUPPORTED_EXTS):
                    files.append(fp)

        out = [f for f in files if not cls._is_own_output(f)]
        out.sort(key=lambda x: natural_key(os.path.basename(x)))
        return out

    @classmethod
    def _is_own_output(cls, path):
        stem, _ = os.path.splitext(os.path.basename(path))
        return any(stem.endswith(s) for s in cls.OUTPUT_SUFFIXES)

    # ------------------------------------------------------------
    #  保存
    # ------------------------------------------------------------
    @staticmethod
    def _save(result, save_dir):
        """保存三张图到 save_dir；任何一张写失败都会抛出明确异常"""
        if "error" in result:
            return
        ok, msg = ensure_output_dir(save_dir)
        if not ok:
            raise RuntimeError(f"输出目录不可用：{save_dir}（{msg}）")

        stem, _ = os.path.splitext(result["source_name"])
        for tag, key in (("1_original", "original"),
                         ("2_unet_mask", "mask"),
                         ("3_yolo", "annotated")):
            img = result.get(key)
            if img is None:
                continue
            out_path = os.path.join(save_dir, f"{stem}_{tag}.png")
            try:
                # 统一存 PNG，避免原图是 jpg 时质量/通道问题
                if img.mode not in ("RGB", "L"):
                    img = img.convert("RGB")
                img.save(out_path)
            except Exception as e:
                raise RuntimeError(f"保存失败 {out_path}: {e}") from e


# ----------------------------------------------------------------
#  工具函数
# ----------------------------------------------------------------
def ensure_output_dir(path):
    """
    确保输出目录存在且可写。
    :return: (ok: bool, msg: str)
    """
    if not path or not str(path).strip():
        return False, "路径为空"
    path = os.path.abspath(str(path).strip().strip('"').strip("'"))
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        return False, f"无法创建目录（{type(e).__name__}: {e}）"
    if not os.path.isdir(path):
        return False, "该路径已存在但是一个文件，不是目录"
    # 可写性实测：写一个临时文件再删掉
    probe = os.path.join(path, ".__write_probe__.tmp")
    try:
        with open(probe, "w", encoding="utf-8") as f:
            f.write("ok")
    except Exception as e:
        return False, f"目录不可写（{type(e).__name__}: {e}）"
    finally:
        try:
            if os.path.exists(probe):
                os.remove(probe)
        except Exception:
            pass  # 删除失败不影响「可写」结论，避免留下报错
    return True, ""


def natural_key(s):
    """自然排序：ceshi-2 排在 ceshi-10 前面"""
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def summarize(results):
    """统计一批结果：总数、成功/失败、各类别检测数、平均耗时"""
    ok = [r for r in results if "error" not in r]
    fail = [r for r in results if "error" in r]
    per_class = {}
    for r in ok:
        for d in r.get("detections", []):
            c = str(d.get("class", "?"))
            per_class[c] = per_class.get(c, 0) + 1
    elapsed = sum(r.get("elapsed", 0.0) for r in ok)
    return {
        "total": len(results),
        "ok": len(ok),
        "fail": len(fail),
        "per_class": per_class,
        "detections": sum(per_class.values()),
        "elapsed": elapsed,
    }


def write_summary_csv(results, save_dir):
    """
    把每张图（每条鱼）的检测统计写成 CSV，方便后续做椎骨计数分析。
    每一行对应一张图，分别列出 chine / coccyx / cocc 等各部位数量。
    """
    path = os.path.join(save_dir, "检测统计.csv")
    names = sorted({str(d.get("class", "?"))
                    for r in results if "error" not in r
                    for d in r.get("detections", [])})

    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["图片", "状态", "检测总数"] + [f"{n}数量" for n in names] +
                   ["平均置信度", "耗时(秒)", "错误信息"])
        for r in results:
            if "error" in r:
                w.writerow([r.get("source_name", ""), "失败", "", *([""] * len(names)),
                            "", "", r.get("error", "")])
                continue
            dets = r.get("detections", [])
            cnt = {n: 0 for n in names}
            for d in dets:
                cnt[str(d.get("class", "?"))] = cnt.get(str(d.get("class", "?")), 0) + 1
            avg = sum(d["conf"] for d in dets) / len(dets) if dets else 0
            w.writerow([r.get("source_name", ""), "成功", len(dets)] +
                       [cnt.get(n, 0) for n in names] +
                       [f"{avg:.4f}", f"{r.get('elapsed', 0):.2f}", ""])
    return path


def default_output_dir():
    """
    默认输出目录：必须是「一定可写」的稳定位置。
    - 打包成 exe 后，程序目录是临时解压目录 _MEIxxxx（退出即删），绝不能往里写结果。
    - 因此 frozen 模式统一写到 文档/黄鳝椎骨识别结果。
    """
    import sys
    if getattr(sys, "frozen", False):
        base = os.path.join(os.path.expanduser("~"), "Documents")
        if not os.path.isdir(base):
            base = os.path.join(os.path.expanduser("~"), "Desktop")
        return os.path.join(base, "黄鳝椎骨识别结果")
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"
    )
