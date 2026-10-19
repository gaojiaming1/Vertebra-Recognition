# -*- coding: utf-8 -*-
"""
启动入口（PyCharm 直接运行本文件）：

    开屏页（脊椎动画 + 模型加载进度） → 加载完成后淡出 → 主界面

模型在开屏阶段就后台加载好，进入主界面后点「开始处理」无需等待。
"""
import os
import sys
import traceback

# 让 PyCharm 运行时能找到 core/ 和 gui/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 【必须最先装】崩溃日志 + 全局异常钩子。
# exe 是 GUI 子系统（console=False），没有它一旦出错就是「闪退、啥也看不见」。
from core.crashlog import install_logging, install_excepthook, log_path

install_logging()
install_excepthook()

from PyQt5.QtCore import QThread, pyqtSignal, QEventLoop, QTimer
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtWidgets import QApplication

from gui.splash import SplashScreen
from gui.theme import APP_QSS


def app_icon_path():
    """
    应用图标位置：
      - 打包后 → _MEIPASS/assets/app.ico（由 spec 的 datas 打进去）
      - 开发模式 → 项目根 assets/app.ico
    图标用 tools/make_icon.py 生成。
    """
    if getattr(sys, "frozen", False):
        p = os.path.join(sys._MEIPASS, "assets", "app.ico")
        if os.path.exists(p):
            return p
    base = os.path.dirname(os.path.abspath(__file__))
    p = os.path.join(base, "assets", "app.ico")
    return p if os.path.exists(p) else ""


class _ModelLoader(QThread):
    """后台加载两个模型（UNet + YOLOv5）"""

    status = pyqtSignal(str)
    done = pyqtSignal(object)
    failed = pyqtSignal(str)

    def run(self):
        try:
            self.status.emit("正在加载分割模型（UNet）…")
            from core.unet_runner import UnetRunner
            from core.yolo_runner import YoloRunner
            from core import Pipeline

            u = UnetRunner.get_default_unet_path()
            y = YoloRunner.get_default_yolo_path()
            if not u or not os.path.exists(u):
                raise FileNotFoundError(f"未找到内置 UNet 权重：{u}")
            if not y or not os.path.exists(y):
                raise FileNotFoundError(f"未找到内置 YOLOv5 权重：{y}")

            self.status.emit("正在加载检测模型（YOLOv5）…")
            pipe = Pipeline(
                unet_path=u, yolo_path=y,
                unet_kwargs={"mix_type": 0},
                yolo_kwargs={"conf_thres": 0.25, "iou_thres": 0.45},
            )
            self.status.emit("模型加载完成")
            self.done.emit(pipe)
        except Exception as e:
            self.failed.emit(f"{e}\n\n{traceback.format_exc()}")


def run_app():
    if sys.platform == "win32":
        # 让任务栏显示独立图标而不是 Python 图标
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(
                "EelVertebra.UNetYOLOPipeline.v2")
        except Exception:
            pass

    print(f"[启动] 日志文件：{log_path()}")

    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    app.setStyleSheet(APP_QSS)

    # 应用图标：任务栏 / 窗口左上角 / Alt-Tab 都用这一个
    ico = app_icon_path()
    if ico:
        app.setWindowIcon(QIcon(ico))
    else:
        print("[启动] 未找到 assets/app.ico，可运行 python tools/make_icon.py 生成")

    # ---------- 开屏页 ----------
    splash = SplashScreen(version="v2.0")
    splash.center_on_screen()
    splash.show()
    splash.set_target(8, "正在初始化…")
    app.processEvents()

    from gui.main_window import MainWindow

    loaded = {"pipe": None, "err": None}

    loader = _ModelLoader()
    loader.status.connect(lambda s: splash.set_target(splash.value(), s))

    def _on_done(pipe):
        loaded["pipe"] = pipe

    def _on_fail(msg):
        loaded["err"] = msg
        splash.set_target(splash.value(), "模型加载失败，进入后可重试")

    loader.done.connect(_on_done)
    loader.failed.connect(_on_fail)
    loader.start()

    # 进度条动画：加载期间缓慢爬升到 88%
    prog = {"v": 10.0}

    def _tick():
        if not loader.isFinished():
            prog["v"] = min(88.0, prog["v"] + 2.6)
            splash.set_target(prog["v"])

    timer = QTimer()
    timer.timeout.connect(_tick)
    timer.start(260)

    wait = QEventLoop()
    loader.finished.connect(wait.quit)
    wait.exec_()
    timer.stop()

    if loaded["err"]:
        print("[启动] 模型加载失败：\n" + loaded["err"])

    win = MainWindow(pipeline=loaded["pipe"])

    def _show_main():
        win.show()
        if loaded["pipe"] is None:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(
                win, "模型未加载",
                "内置模型加载失败，处理时会自动重试。\n\n"
                f"错误信息：{str(loaded['err'])[:400]}"
            )

    splash.finish(_show_main)

    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()
