# -*- coding: utf-8 -*-
"""
崩溃/日志工具。

打包成 exe 后是 GUI 子系统（console=False），print / traceback 全都看不见，
一出错就是「双击 → 黑影一闪 → 没了」，完全没法排查。

这里做两件事：
  1. install_logging()：把 stdout / stderr 同时写到一个固定位置的日志文件
     （存放位置：%LOCALAPPDATA%\\EelVertebra\\app.log，开发模式写到项目根目录 app.log）
  2. install_excepthook()：捕获所有未处理异常 → 写日志 + 弹窗显示错误摘要，
     让「闪退」变成「看得见的报错」
"""
import os
import sys
import traceback
from datetime import datetime


def log_dir():
    base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
    return os.path.join(base, "EelVertebra")


def log_path():
    return os.path.join(log_dir(), "app.log")


class _Tee:
    """把输出同时写到终端和日志文件"""

    def __init__(self, stream, file_obj):
        self._stream = stream
        self._file = file_obj

    def write(self, data):
        try:
            self._stream.write(data)
        except Exception:
            pass
        try:
            self._file.write(data)
            self._file.flush()
        except Exception:
            pass

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass
        try:
            self._file.flush()
        except Exception:
            pass

    def isatty(self):
        return False

    # 未知属性一律转交给原始 stream（fileno / encoding / buffer …），
    # 避免第三方库（tqdm、torch 的日志判断等）因为缺少属性而报错
    def __getattr__(self, name):
        if self._stream is None:
            raise AttributeError(name)
        return getattr(self._stream, name)


def install_logging():
    """把 stdout/stderr 复制一份到日志文件，正常返回 True，失败不影响程序启动"""
    try:
        os.makedirs(log_dir(), exist_ok=True)
        f = open(log_path(), "a", encoding="utf-8", errors="replace")
        f.write(f"\n\n===== {datetime.now():%Y-%m-%d %H:%M:%S} 程序启动 "
                f"(frozen={getattr(sys, 'frozen', False)}) =====\n")
        f.flush()
        sys.stdout = _Tee(sys.stdout, f)
        sys.stderr = _Tee(sys.stderr, f)
        return True
    except Exception as e:
        print(f"[crashlog] 日志文件打开失败：{e}")
        return False


def install_excepthook():
    """安装全局异常处理：写日志 + 弹窗，避免 exe 静默闪退"""
    def _hook(exc_type, exc_value, exc_tb):
        text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        try:
            with open(log_path(), "a", encoding="utf-8", errors="replace") as f:
                f.write(f"\n[未捕获异常] {datetime.now():%Y-%m-%d %H:%M:%S}\n{text}\n")
        except Exception:
            pass
        sys.__stderr__.write(text) if hasattr(sys, "__stderr__") else None
        try:
            from PyQt5.QtWidgets import QMessageBox, QApplication
            app = QApplication.instance()
            if app is not None:
                short = f"{exc_type.__name__}: {exc_value}"
                QMessageBox.critical(
                    None, "程序出错",
                    f"{short}\n\n详细堆栈已写入：\n{log_path()}"
                )
        except Exception:
            pass
        # 最后还是走系统默认行为
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _hook
