# -*- coding: utf-8 -*-
"""
统一视觉主题（黄鳝椎骨识别 · 清水青竹配色）

配色思路：
  主色 竹青 #0E9488 / 深潭 #0F5C57 —— 呼应「黄鳝 · 水田 · 骨骼」的意象
  背景 米白 #F5F7F8，卡片纯白 + 1px 淡边框 + 圆角，整体干净克制
"""
from PyQt5.QtGui import QColor

# ---- 主色板 ----
C_PRIMARY = "#0E9488"        # 竹青（主按钮 / 强调）
C_PRIMARY_DARK = "#0B7268"   # 主色按下
C_PRIMARY_LIGHT = "#E6F4F2"  # 主色浅底
C_ACCENT = "#F2A03D"         # 琥珀（数量标记）
C_DANGER = "#D9534F"         # 停止按钮
C_TEXT = "#1F2D33"
C_TEXT_SUB = "#6B7F86"
C_BG = "#F5F7F8"
C_CARD = "#FFFFFF"
C_BORDER = "#E3EAEC"
C_HEAD_TOP = "#0F766E"
C_HEAD_BOTTOM = "#0A4F4A"

# ---- 类别配色（与 yolo_runner 绘制框一致：chine 红橙黄 / coccyx 绿 / cocc 品红紫）----
CLASS_COLORS = {
    "chine": "#E24A4A",
    "coccyx": "#3FAE5A",
    "cocc": "#B44BC9",
}
CLASS_LABELS = {
    "chine": "chine 椎骨",
    "coccyx": "coccyx 尾椎",
    "cocc": "cocc 尾端",
}


def class_color(name, fallback="#8A9BA3"):
    return CLASS_COLORS.get(str(name).lower(), fallback)


def qcolor(hex_str):
    return QColor(hex_str)


APP_QSS = """
/* ============ 全局 ============ */
QWidget {
    font-family: "Microsoft YaHei", "PingFang SC", "Segoe UI";
    font-size: 13px;
    color: #1F2D33;
}
QMainWindow, QWidget#page {
    background: #F5F7F8;
}
QToolTip {
    background: #1F2D33; color: #fff; border: none; padding: 4px 8px;
}

/* ============ 卡片 ============ */
QFrame#card {
    background: #FFFFFF;
    border: 1px solid #E3EAEC;
    border-radius: 10px;
}
/* 命名式透明容器：只对自身生效，不污染子控件样式 */
QWidget#transparentPanel {
    background: transparent;
    border: none;
}
QLabel#cardTitle {
    color: #0F5C57;
    font-size: 13px;
    font-weight: bold;
}
QLabel#hint {
    color: #8A9BA3;
    font-size: 11px;
}
QLabel#chip {
    background: #E6F4F2;
    color: #0B7268;
    border-radius: 9px;
    padding: 2px 9px;
    font-size: 11px;
}

/* ============ 头部 ============ */
QFrame#header {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #0F766E, stop:1 #0A4F4A);
    border: none;
}
QLabel#appTitle {
    color: #FFFFFF; font-size: 19px; font-weight: bold; letter-spacing: 1px;
}
QLabel#appSub {
    color: rgba(255,255,255,0.72); font-size: 11px;
}
QLabel#verChip {
    color: #FFFFFF; font-size: 11px;
    background: rgba(255,255,255,0.16);
    border-radius: 10px; padding: 3px 10px;
}

/* ============ 输入控件 ============ */
QLineEdit {
    background: #FFFFFF;
    border: 1px solid #DCE3E6;
    border-radius: 6px;
    padding: 7px 9px;
    font-size: 12px;
    selection-background-color: #0E9488;
}
QLineEdit:focus { border: 1px solid #0E9488; }
QLineEdit:read-only { background: #FAFCFC; color: #4C5F66; }

QComboBox {
    background: #FFFFFF;
    border: 1px solid #DCE3E6;
    border-radius: 6px;
    padding: 6px 9px;
    font-size: 12px;
    min-height: 18px;
}
QComboBox:focus { border: 1px solid #0E9488; }
QComboBox::drop-down { border: none; width: 20px; }
QComboBox QAbstractItemView {
    border: 1px solid #DCE3E6; background: #fff;
    selection-background-color: #E6F4F2; selection-color: #0B7268;
}

QDoubleSpinBox, QSpinBox {
    background: #FFFFFF;
    border: 1px solid #DCE3E6;
    border-radius: 6px;
    padding: 5px 6px;
    font-size: 12px;
}
QDoubleSpinBox:focus, QSpinBox:focus { border: 1px solid #0E9488; }

/* ============ 按钮 ============ */
QPushButton {
    background: #FFFFFF;
    border: 1px solid #DCE3E6;
    border-radius: 6px;
    padding: 7px 14px;
    font-size: 12px;
    color: #2B3B42;
}
QPushButton:hover { background: #F0F5F5; border: 1px solid #B9CFCC; }
QPushButton:pressed { background: #E4EEED; }
QPushButton:disabled { background: #F2F5F6; color: #A9B6BB; border: 1px solid #E3EAEC; }

QPushButton#primary {
    background: #0E9488; border: none; color: #FFFFFF;
    font-size: 14px; font-weight: bold; padding: 11px 16px; border-radius: 8px;
}
QPushButton#primary:hover { background: #0B7F74; }
QPushButton#primary:pressed { background: #0B7268; }
QPushButton#primary:disabled { background: #9FC9C4; color: #FFFFFF; }

QPushButton#stop {
    background: #FFFFFF; border: 1px solid #E7B4B2; color: #C3504C;
    font-size: 13px; padding: 10px 14px; border-radius: 8px;
}
QPushButton#stop:hover { background: #FDF2F1; }
QPushButton#stop:disabled { background: #F7F9F9; color: #BFCACB; border: 1px solid #E3EAEC; }

QPushButton#ghost {
    background: transparent; border: 1px dashed #C6D3D6; color: #0B7268;
}
QPushButton#ghost:hover { background: #E6F4F2; border: 1px solid #0E9488; }

/* ============ 滑块 ============ */
QSlider::groove:horizontal {
    background: #E3EAEC; height: 4px; border-radius: 2px;
}
QSlider::sub-page:horizontal {
    background: #0E9488; height: 4px; border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #FFFFFF; border: 2px solid #0E9488;
    width: 14px; margin: -6px 0; border-radius: 8px;
}
QSlider::handle:horizontal:hover { background: #E6F4F2; }

/* ============ 进度条 ============ */
QProgressBar {
    background: #EDF1F2; border: none; border-radius: 5px;
    height: 12px; text-align: center; color: #6B7F86; font-size: 10px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #0E9488, stop:1 #35B6A6);
    border-radius: 5px;
}

/* ============ 分组 & 文本区 ============ */
QGroupBox {
    border: 1px solid #E3EAEC; border-radius: 8px; margin-top: 12px;
    background: #FFFFFF; font-weight: bold;
}
QGroupBox::title {
    subcontrol-origin: margin; left: 12px; padding: 0 6px; color: #0F5C57;
}
QTextEdit {
    background: #FCFDFD; border: 1px solid #E3EAEC; border-radius: 6px;
    font-family: Consolas, "Microsoft YaHei"; font-size: 11px; color: #40535A;
}
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #F0F3F4; width: 9px; margin: 0; border: none; }
QScrollBar::handle:vertical { background: #CBD6D9; border-radius: 4px; min-height: 24px; }
QScrollBar::handle:vertical:hover { background: #AEBEC2; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: #F0F3F4; height: 9px; border: none; }
QScrollBar::handle:horizontal { background: #CBD6D9; border-radius: 4px; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
"""
