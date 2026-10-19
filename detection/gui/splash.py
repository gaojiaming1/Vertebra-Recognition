# -*- coding: utf-8 -*-
"""
开屏页面（Splash Screen）——黄鳝椎骨智能识别系统

- 无边框圆角窗口 + 深潭青渐变底
- QPainter 手绘「脊椎骨」装饰图形（椎体 + 棘突），呼应椎骨识别主题
- 进度条平滑动画：外部只需 set_target(value, status)
- 收尾用 windowOpacity 动画淡出
"""
import math

from PyQt5.QtCore import Qt, QTimer, QRectF, QPointF, QPropertyAnimation, pyqtSignal
from PyQt5.QtGui import (
    QPainter, QColor, QFont, QLinearGradient, QPen, QBrush, QPainterPath, QPolygonF,
)
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QProgressBar, QApplication,
)


# 开屏页只显示这一行标题（+ 版本号），不再展示任何技术栈说明
SPLASH_TITLE = "基于CT图像的鱼类脊椎骨识别与分析软件"
SPLASH_VERSION = "V2.0"


class SplashScreen(QWidget):
    """开屏页：显示品牌信息 + 模型加载进度"""

    W, H = 780, 440

    def __init__(self, version=SPLASH_VERSION, title=SPLASH_TITLE):
        super().__init__()
        self.version = version
        self.title_text = title
        self._value = 0.0
        self._target = 0.0

        self.setWindowTitle("启动中")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setFixedSize(self.W, self.H)

        self._build_ui()

        # 平滑动画：每 30ms 向目标值靠近
        self._timer = QTimer(self)
        self._timer.setInterval(30)
        self._timer.timeout.connect(self._tick)
        self._timer.start()

    # ---------------- UI ----------------
    def _build_ui(self):
        lay = QVBoxLayout(self)
        lay.setContentsMargins(56, 200, 56, 40)
        lay.setSpacing(0)

        # 标题 + 版本号（同一行）
        title_row = QHBoxLayout()
        title_row.setSpacing(12)
        self.title = QLabel(self.title_text)
        self.title.setStyleSheet(
            "color:#FFFFFF; font-size:26px; font-weight:bold; letter-spacing:2px; "
            "background:transparent;"
        )
        title_row.addWidget(self.title)
        ver = QLabel(self.version)
        ver.setStyleSheet(
            "color:#0A5550; background:#7FE9DA; border-radius:9px; "
            "padding:3px 10px; font-size:13px; font-weight:bold;"
        )
        title_row.addWidget(ver, 0, Qt.AlignBottom)
        title_row.addStretch(1)
        lay.addLayout(title_row)

        lay.addSpacing(52)

        self.status = QLabel("正在启动…")
        self.status.setStyleSheet(
            "color:rgba(255,255,255,0.62); font-size:11px; background:transparent;"
        )
        lay.addWidget(self.status)
        lay.addSpacing(7)

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setValue(0)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(5)
        self.bar.setStyleSheet(
            """
            QProgressBar { background: rgba(255,255,255,0.16); border: none; border-radius: 3px; }
            QProgressBar::chunk {
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                            stop:0 #4FD1C5, stop:1 #F2A03D);
                border-radius: 3px;
            }
            """
        )
        lay.addWidget(self.bar)

        lay.addSpacing(14)

        self.pct = QLabel("0%")
        self.pct.setStyleSheet(
            "color:rgba(255,255,255,0.55); font-size:11px; background:transparent;"
        )
        lay.addWidget(self.pct, 0, Qt.AlignRight)

        lay.addStretch(1)

    # ---------------- 对外接口 ----------------
    def set_target(self, value, status=None):
        """设置目标进度（0~100），进度条会平滑追上去"""
        self._target = max(self._target, float(value))
        if status:
            self.status.setText(status)

    def value(self):
        return self._value

    def _tick(self):
        if abs(self._value - self._target) < 0.3:
            self._value = self._target
        else:
            self._value += (self._target - self._value) * 0.18
        v = int(self._value)
        if self.bar.value() != v:
            self.bar.setValue(v)
            self.pct.setText(f"{v}%")
        self.update()  # 触发重绘（脊椎高亮随进度点亮）

    def finish(self, on_done):
        """加载完成后淡出，淡出结束回调 on_done"""
        self._timer.stop()
        self.set_target(100, "准备就绪")
        self._tick()

        def _fade():
            anim = QPropertyAnimation(self, b"windowOpacity")
            anim.setDuration(420)
            anim.setStartValue(1.0)
            anim.setEndValue(0.0)
            anim.finished.connect(lambda: (self.close(), on_done()))
            anim.start()
            self._anim = anim  # 持有引用，避免被 GC

        QTimer.singleShot(260, _fade)

    # ---------------- 绘制 ----------------
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)

        # 圆角背景
        path = QPainterPath()
        path.addRoundedRect(QRectF(0, 0, self.W, self.H), 16, 16)
        grad = QLinearGradient(0, 0, self.W, self.H)
        grad.setColorAt(0.0, QColor("#0E7C70"))
        grad.setColorAt(0.55, QColor("#0A5550"))
        grad.setColorAt(1.0, QColor("#063B38"))
        p.fillPath(path, QBrush(grad))

        # 右上角柔光
        glow = QLinearGradient(self.W * 0.55, 0, self.W, self.H * 0.6)
        glow.setColorAt(0.0, QColor(255, 255, 255, 0))
        glow.setColorAt(1.0, QColor(120, 230, 210, 42))
        p.fillPath(path, QBrush(glow))

        # 脊椎装饰
        self._draw_spine(p)

        super().paintEvent(event)

    def _draw_spine(self, p):
        """画一串脊椎骨：椎体（椭圆）+ 棘突（三角），随进度逐个点亮"""
        n = 16
        x0, x1 = 62, self.W - 62
        base_y = 118.0
        lit = int(self._value / 100.0 * n + 0.0001)

        for i in range(n):
            t = i / (n - 1.0)
            x = x0 + (x1 - x0) * t
            y = base_y + 26.0 * math.sin(t * math.pi * 1.6) - 8.0 * math.cos(t * math.pi * 3.0)
            scale = 0.72 + 0.34 * math.sin(t * math.pi)      # 中段椎骨更大
            rx, ry = 12.0 * scale, 8.5 * scale

            is_lit = i < lit
            alpha = 235 if is_lit else 62
            body = QColor(214, 255, 246, alpha)
            line = QColor(9, 60, 56, 150 if is_lit else 40)

            # 棘突（向上）
            proc = QPolygonF([
                QPointF(x - 3.0 * scale, y - ry + 1),
                QPointF(x, y - ry - 15.0 * scale),
                QPointF(x + 3.0 * scale, y - ry + 1),
            ])
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(body))
            p.drawPolygon(proc)

            # 椎体
            p.setPen(QPen(line, 1.2))
            p.setBrush(QBrush(body))
            p.drawEllipse(QPointF(x, y), rx, ry)

            # 椎体中心小孔（椎孔）
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(6, 59, 56, 110 if is_lit else 30)))
            p.drawEllipse(QPointF(x, y), rx * 0.32, ry * 0.32)

            # 椎间隙
            if i < n - 1:
                tx = x0 + (x1 - x0) * ((i + 1) / (n - 1.0))
                p.setPen(QPen(QColor(255, 255, 255, 70 if is_lit else 28), 1.6))
                p.drawLine(QPointF(x + rx, y), QPointF(tx - rx * 0.9, y))

    # 屏幕居中
    def center_on_screen(self):
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            self.move(geo.center().x() - self.W // 2, geo.center().y() - self.H // 2)
