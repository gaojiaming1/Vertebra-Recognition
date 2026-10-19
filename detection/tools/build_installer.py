# -*- coding: utf-8 -*-
"""
准备「安装包」文件夹：
    dist/安装包/
        ├─ 鱼类脊椎骨识别_V2.0_安装程序.exe   （35 MB，双击运行）
        └─ unet_yolo_pipeline.exe              （主程序，约 1.9 GB，安装源）

说明：
  PyInstaller 单文件 exe 的启动引导程序要在**文件末尾**找自己的归档标记，
  所以不能像 7z 自解压那样把主程序追加到安装程序尾部（会导致安装程序启动不了）。
  因此采用「安装程序 + 同目录主程序」的两个文件形式：
  两个文件放同一个文件夹，双击安装程序即可选择路径安装，并创建快捷方式/卸载项。
  安装程序找不到主程序时会弹窗让用户手动选择。

用法：
    python tools/build_installer.py
"""
import os
import shutil

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORE_NAME = "鱼类脊椎骨识别_V2.0_安装程序.exe"
CORE_SRC = os.path.join(BASE, "installer", "dist", CORE_NAME)
APP_SRC = os.path.join(BASE, "dist", "鱼类脊椎骨识别_V2.0.exe")
OUT_DIR = os.path.join(BASE, "dist", "安装包")


def _copy(src, dst):
    print("  复制 %s -> %s" % (os.path.basename(src), dst))
    shutil.copy2(src, dst)


def main():
    for p, what in ((CORE_SRC, "安装向导"), (APP_SRC, "主程序")):
        if not os.path.exists(p):
            raise SystemExit("缺少%s：%s" % (what, p))

    os.makedirs(OUT_DIR, exist_ok=True)
    print("准备安装包目录：%s" % OUT_DIR)
    _copy(CORE_SRC, os.path.join(OUT_DIR, CORE_NAME))
    _copy(APP_SRC, os.path.join(OUT_DIR, "unet_yolo_pipeline.exe"))

    core_sz = os.path.getsize(CORE_SRC)
    app_sz = os.path.getsize(APP_SRC)
    print()
    print("✅ 安装包已准备好：%s" % OUT_DIR)
    print("   %-32s %6.0f MB" % (CORE_NAME, core_sz / 1048576))
    print("   %-32s %6.0f MB" % ("unet_yolo_pipeline.exe", app_sz / 1048576))
    print()
    print("使用：把整个「安装包」文件夹拷给对方 → 双击「%s」→ 选路径 → 安装" % CORE_NAME)
    print("（安装程序会自动识别同目录的主程序；若只拷了安装程序，运行时会让你手动选择主程序）")


if __name__ == "__main__":
    main()
