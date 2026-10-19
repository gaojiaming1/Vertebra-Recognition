# -*- coding: utf-8 -*-
"""稳健删除目录：处理 Windows 长路径(>260)与只读属性。"""
import os
import sys
import stat
import shutil


def _on_error(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        # 长路径：加 \\?\ 前缀重试
        try:
            lp = u"\\\\?\\" + os.path.abspath(path)
            os.chmod(lp, stat.S_IWRITE)
            func(lp)
        except Exception:
            pass


def force_remove(path):
    if not os.path.exists(path):
        print("  跳过(不存在):", path)
        return
    # 先把所有文件/子目录去掉只读
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            p = os.path.join(root, name)
            try:
                os.chmod(p, stat.S_IWRITE)
            except Exception:
                pass
    lp = u"\\\\?\\" + os.path.abspath(path)
    try:
        shutil.rmtree(lp, onerror=_on_error)
    except Exception:
        shutil.rmtree(path, onerror=_on_error)
    print("  已删:", path, "->", "OK" if not os.path.exists(path) else "仍残留")


if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    project = os.path.dirname(base)
    targets = sys.argv[1:] or [
        os.path.join(project, "build_v5"),
        os.path.join(project, "build_v6"),
        os.path.join(project, "dist_v4"),
        os.path.join(project, "dist_v5"),
        os.path.join(project, "dist_v6"),
        os.path.join(project, "build"),
        os.path.join(project, "dist"),
    ]
    for t in targets:
        force_remove(t)
    print("清理完成。")
