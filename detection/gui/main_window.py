# -*- coding: utf-8 -*-
"""
主窗口 —— 黄鳝椎骨智能识别系统

布局：
    ┌───────────────────────────────────────────────────────────┐
    │  头部（渐变）：标题 / 副标题 / 版本                          │
    ├──────────────┬────────────────────────────────────────────┤
    │ 输入卡片      │  ① 原图  ② UNet 分割  ③ YOLOv5 检测         │
    │ 输出卡片      │  ─────────────────────────────             │
    │ 参数卡片      │  统计 chips + 进度条 + 日志                  │
    │ 开始 / 停止   │                                            │
    └──────────────┴────────────────────────────────────────────┘

说明：
  - 权重已内置（UNet + YOLOv5），界面不再暴露权重选择，只保留 置信度 / IoU 两个阈值。
  - 输出目录可自由选择（支持在对话框里新建文件夹），开始前会做「可写性预检」，
    写不进去会立刻提示原因，而不是跑完才发现没存下。
"""
import os
import sys
import time
import traceback
from datetime import datetime

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QSettings, QTimer, QPointF
from PyQt5.QtGui import QPixmap, QImage, QFont, QColor, QPainter
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QComboBox,
    QProgressBar, QTextEdit, QGridLayout, QFrame, QDoubleSpinBox,
    QMessageBox, QSplitter, QStatusBar, QSlider, QCheckBox, QScrollArea,
    QDialog, QSizePolicy, QDesktopWidget, QHeaderView, QAbstractItemView,
    QTableWidget, QTableWidgetItem,
)

from core import Pipeline
from core.pipeline import ensure_output_dir, default_output_dir, summarize
from .theme import APP_QSS, CLASS_COLORS, CLASS_LABELS


# ---------------- 子线程：跑推理 ----------------
class InferenceWorker(QThread):
    """推理 worker，跑在子线程，避免阻塞 GUI"""

    progress = pyqtSignal(int, int, object)   # done, total, last_result
    log = pyqtSignal(str)
    finished_ok = pyqtSignal(list)            # all results
    error = pyqtSignal(str)

    def __init__(self, inputs, save_dir, conf_thres, iou_thres,
                 pipeline=None, unet_path=None, yolo_path=None):
        super().__init__()
        self.inputs = inputs
        self.save_dir = save_dir
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.pipeline = pipeline              # 开屏阶段预加载好的（可为 None）
        self.unet_path = unet_path
        self.yolo_path = yolo_path
        self._stop = False

    def stop(self):
        self._stop = True

    def _ensure_pipeline(self):
        if self.pipeline is None:
            from core.unet_runner import UnetRunner
            from core.yolo_runner import YoloRunner
            u = self.unet_path or UnetRunner.get_default_unet_path()
            y = self.yolo_path or YoloRunner.get_default_yolo_path()
            self.log.emit(f"[{self._now()}] 加载内置模型…")
            self.pipeline = Pipeline(
                unet_path=u, yolo_path=y,
                unet_kwargs={"mix_type": 0},
                yolo_kwargs={"conf_thres": self.conf_thres, "iou_thres": self.iou_thres},
            )
            self.log.emit(f"[{self._now()}] 模型加载完成")
        # 复用模型时同步阈值（不重新加载权重）
        self.pipeline.set_params(conf_thres=self.conf_thres, iou_thres=self.iou_thres,
                                 mix_type=0)

    @staticmethod
    def _now():
        return f"{datetime.now():%H:%M:%S}"

    def run(self):
        try:
            self._ensure_pipeline()
            self.log.emit(f"[{self._now()}] 开始处理，共 {len(self.inputs)} 张图")

            def cb(done, total, last):
                if self._stop:
                    return
                self.progress.emit(done, total, last)
                if isinstance(last, dict) and "error" in last:
                    self.log.emit(
                        f"[{self._now()}] ❌ {last.get('source_name')} 失败："
                        f"{last.get('error')}"
                    )

            results = self.pipeline.run_batch(self.inputs, save_dir=self.save_dir,
                                              progress_cb=cb)
            if self._stop:
                self.log.emit(f"[{self._now()}] 用户中止")
            self.finished_ok.emit(results)
        except Exception as e:
            tb = traceback.format_exc()
            self.error.emit(f"{e}\n\n{tb}")


# ---------------- 图像显示卡片 ----------------
class ImagePanel(QFrame):
    """带标题的图片显示卡片（双击可放大查看）"""

    def __init__(self, title, accent="#0E9488"):
        super().__init__()
        self.setObjectName("card")
        self._pixmap = None
        self.accent = accent

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 8, 10, 10)
        lay.setSpacing(6)

        head = QHBoxLayout()
        self.dot = QLabel()
        self.dot.setFixedSize(9, 9)
        self.dot.setStyleSheet(
            f"background:{accent}; border-radius:4px; border:none;"
        )
        head.addWidget(self.dot)
        self.title_label = QLabel(title)
        self.title_label.setObjectName("cardTitle")
        head.addWidget(self.title_label)
        head.addStretch(1)
        lay.addLayout(head)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(240, 240)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.image_label.setStyleSheet(
            "QLabel { background:#FAFCFC; border:1px dashed #DCE3E6; border-radius:6px; "
            "color:#A9B6BB; font-size:12px; }"
        )
        self.image_label.setText("等待图片")
        self.image_label.setScaledContents(False)
        lay.addWidget(self.image_label, 1)

    def set_title(self, t):
        self.title_label.setText(t)

    def show_pil(self, pil_image):
        if pil_image is None:
            return
        img = pil_image.convert("RGB")
        data = img.tobytes("raw", "RGB")
        qimg = QImage(data, img.width, img.height, img.width * 3, QImage.Format_RGB888)
        self._pixmap = QPixmap.fromImage(qimg)
        self._rescale()

    def _rescale(self):
        if self._pixmap is None:
            return
        scaled = self._pixmap.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()

    def mouseDoubleClickEvent(self, event):
        if self._pixmap is None:
            return
        dlg = QDialog(self)
        dlg.setWindowTitle(self.title_label.text())
        dlg.setStyleSheet("background:#1B262A;")
        lay = QVBoxLayout(dlg)
        lab = QLabel()
        lab.setAlignment(Qt.AlignCenter)
        screen = QDesktopWidget().availableGeometry(self)
        pm = self._pixmap.scaled(
            QSize(int(screen.width() * 0.9), int(screen.height() * 0.85)),
            Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        lab.setPixmap(pm)
        lay.addWidget(lab)
        dlg.exec_()

    def clear(self, text="等待图片"):
        self._pixmap = None
        self.image_label.setText(text)
        self.image_label.setPixmap(QPixmap())


# ---------------- 小组件 ----------------
def _card(title, hint=None):
    box = QFrame()
    box.setObjectName("card")
    lay = QVBoxLayout(box)
    lay.setContentsMargins(14, 10, 14, 12)
    lay.setSpacing(8)
    t = QLabel(title)
    t.setObjectName("cardTitle")
    lay.addWidget(t)
    if hint:
        h = QLabel(hint)
        h.setObjectName("hint")
        h.setWordWrap(True)
        lay.addWidget(h)
    return box, lay


class StatChip(QLabel):
    """统计小胶囊：● chine 96"""

    def __init__(self, name, color):
        super().__init__()
        self.setObjectName("chip")
        self._name = name
        self.set_value(0)
        # 注意：Qt QSS 只认 #AARRGGBB / rgba()，不认 #RRGGBBAA
        c = QColor(color)
        r, g, b = c.red(), c.green(), c.blue()
        self.setStyleSheet(
            f"QLabel#chip {{ background: rgba({r},{g},{b},0.10); color:{color}; "
            f"border:1px solid rgba({r},{g},{b},0.35); border-radius:10px; "
            f"padding:3px 10px; font-size:12px; font-weight:bold; }}"
        )

    def set_value(self, v):
        self.setText(f"● {self._name}  {v}")


def count_by_class(detections, classes):
    """按类别统计数量（未出现的类别补 0）"""
    cnt = {k: 0 for k in classes}
    for d in detections or []:
        k = str(d.get("class", "")).lower()
        cnt[k] = cnt.get(k, 0) + 1
    return cnt


# ---------------- 主窗口 ----------------
class MainWindow(QMainWindow):
    def __init__(self, pipeline=None):
        super().__init__()
        self.setWindowTitle("基于CT图像的鱼类脊椎骨识别与分析软件  V2.0")
        # 窗口自适应屏幕：小屏笔记本也不会被任务栏挡住
        screen = QDesktopWidget().availableGeometry()
        self.resize(min(1420, int(screen.width() * 0.94)),
                    min(860, int(screen.height() * 0.94)))
        self.setMinimumSize(1080, 640)

        self.pipeline = pipeline      # 开屏阶段预加载的模型（可能为 None）
        self.worker = None
        self.current_results = []
        self._t0 = 0.0

        self.settings = QSettings("EelVertebra", "UNetYOLOPipeline")

        self._init_ui()
        self._connect_signals()
        self._restore_settings()

        self.statusBar().showMessage("就绪")

    # ================= UI =================
    def _init_ui(self):
        central = QWidget()
        central.setObjectName("page")
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 12, 12)
        root.setSpacing(12)

        root.addWidget(self._build_header())

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self._build_left_panel())
        body.addWidget(self._build_right_area(), 1)
        root.addLayout(body, 1)

        self.setStatusBar(QStatusBar())

    # ---- 头部 ----
    def _build_header(self):
        head = QFrame()
        head.setObjectName("header")
        head.setFixedHeight(74)
        lay = QHBoxLayout(head)
        lay.setContentsMargins(22, 0, 22, 0)

        # 左侧小图标（画一条脊椎）
        icon = QLabel()
        icon.setFixedSize(46, 46)
        pm = QPixmap(46, 46)
        pm.fill(Qt.transparent)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.NoPen)
        for i in range(5):
            x = 6 + i * 8.5
            y = 23 + (i - 2) * 1.6
            r = 4.2 - abs(i - 2) * 0.5
            alpha = 255 - abs(i - 2) * 45
            painter.setBrush(QColor(160, 255, 235, alpha))
            painter.drawEllipse(QPointF(x, y), r, r)
        painter.end()
        icon.setPixmap(pm)
        lay.addWidget(icon)
        lay.addSpacing(6)

        left = QVBoxLayout()
        left.setSpacing(2)
        t = QLabel("基于CT图像的鱼类脊椎骨识别与分析软件")
        t.setObjectName("appTitle")
        s = QLabel("CT 影像 → 椎骨分割与计数（chine / coccyx / cocc）  |  支持单张 / 文件夹批量")
        s.setObjectName("appSub")
        left.addWidget(t)
        left.addWidget(s)
        lay.addLayout(left)
        lay.addStretch(1)

        v = QLabel("V2.0")
        v.setObjectName("verChip")
        lay.addWidget(v, 0, Qt.AlignVCenter)
        return head

    # ---- 左侧控制区 ----
    def _build_left_panel(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(330)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        container = QWidget()
        container.setObjectName("transparentPanel")
        lay = QVBoxLayout(container)
        lay.setContentsMargins(2, 2, 6, 2)
        lay.setSpacing(12)

        # ===== 输入 =====
        box, bl = _card("① 输入")
        row = QHBoxLayout()
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["单张图片", "文件夹批量"])
        row.addWidget(self.mode_combo, 1)
        bl.addLayout(row)

        self.recursive_check = QCheckBox("包含子文件夹（批量模式）")
        self.recursive_check.setStyleSheet("color:#6B7F86; font-size:12px;")
        bl.addWidget(self.recursive_check)

        self.input_path_edit = QLineEdit()
        self.input_path_edit.setPlaceholderText("")
        self.input_path_edit.setReadOnly(True)
        bl.addWidget(self.input_path_edit)

        r2 = QHBoxLayout()
        self.btn_input_browse = QPushButton("选择图片…")
        self.btn_input_browse.setObjectName("ghost")
        r2.addWidget(self.btn_input_browse)
        self.btn_input_clear = QPushButton("清空")
        r2.addWidget(self.btn_input_clear)
        bl.addLayout(r2)

        self.input_info = QLabel("")
        self.input_info.setObjectName("hint")
        bl.addWidget(self.input_info)
        lay.addWidget(box)

        # ===== 输出 =====
        box2, bl2 = _card("② 输出目录", "结果图与统计表会保存到该目录（可新建）")
        self.output_path_edit = QLineEdit()
        self.output_path_edit.setPlaceholderText("")
        bl2.addWidget(self.output_path_edit)

        r3 = QHBoxLayout()
        self.btn_output_browse = QPushButton("选择…")
        self.btn_output_browse.setObjectName("ghost")
        r3.addWidget(self.btn_output_browse)
        self.btn_output_open = QPushButton("打开目录")
        r3.addWidget(self.btn_output_open)
        bl2.addLayout(r3)

        self.output_state = QLabel("")
        self.output_state.setObjectName("hint")
        bl2.addWidget(self.output_state)
        lay.addWidget(box2)

        # ===== 参数 =====
        box3, bl3 = _card("③ 检测参数", "权重已内置，只需调节这两个阈值")
        self._add_slider(bl3, "置信度阈值", "conf", 0.05, 0.95, 0.25)
        self._add_slider(bl3, "IoU 阈值", "iou", 0.10, 0.90, 0.45)
        note = QLabel("置信度↑ 更保守（漏检↑误检↓）；IoU↑ 重叠框合并更激进")
        note.setObjectName("hint")
        note.setWordWrap(True)
        bl3.addWidget(note)
        lay.addWidget(box3)

        # ===== 运行 =====
        run_box = QFrame()
        run_box.setObjectName("transparentPanel")
        rl = QVBoxLayout(run_box)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(8)
        self.btn_start = QPushButton("▶  开始处理")
        self.btn_start.setObjectName("primary")
        self.btn_start.setMinimumHeight(46)
        rl.addWidget(self.btn_start)
        row4 = QHBoxLayout()
        self.btn_stop = QPushButton("■  停止")
        self.btn_stop.setObjectName("stop")
        self.btn_stop.setEnabled(False)
        row4.addWidget(self.btn_stop)
        self.btn_clear_log = QPushButton("清空日志")
        row4.addWidget(self.btn_clear_log)
        rl.addLayout(row4)
        lay.addWidget(run_box)

        lay.addStretch(1)
        scroll.setWidget(container)
        return scroll

    def _add_slider(self, parent, label, key, lo, hi, default):
        """滑块 + 数值框联动的一行"""
        cap = QHBoxLayout()
        lab = QLabel(label)
        lab.setStyleSheet("font-size:12px; color:#40535A;")
        cap.addWidget(lab)
        cap.addStretch(1)
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setValue(default)
        spin.setFixedWidth(66)
        spin.setButtonSymbols(QDoubleSpinBox.NoButtons)
        cap.addWidget(spin)
        parent.addLayout(cap)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(lo * 100), int(hi * 100))
        slider.setValue(int(default * 100))
        slider.setSingleStep(5)
        parent.addWidget(slider)

        def _from_slider(v):
            spin.blockSignals(True)
            spin.setValue(v / 100.0)
            spin.blockSignals(False)

        def _from_spin(v):
            slider.blockSignals(True)
            slider.setValue(int(round(v * 100)))
            slider.blockSignals(False)

        slider.valueChanged.connect(_from_slider)
        spin.valueChanged.connect(_from_spin)

        if key == "conf":
            self.conf_spin, self.conf_slider = spin, slider
        else:
            self.iou_spin, self.iou_slider = spin, slider

    # ---- 右侧图像 + 进度 + 日志 ----
    def _build_right_area(self):
        area = QWidget()
        area.setObjectName("transparentPanel")
        lay = QVBoxLayout(area)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(12)

        mid = QSplitter(Qt.Horizontal)
        mid.setHandleWidth(10)
        self.panel_orig = ImagePanel("① 原图", "#8FA6AE")
        self.panel_mask = ImagePanel("② UNet 分割结果（Blend）", "#3B8FB0")
        self.panel_yolo = ImagePanel("③ YOLOv5 检测结果", "#0E9488")
        mid.addWidget(self.panel_orig)
        mid.addWidget(self.panel_mask)
        mid.addWidget(self.panel_yolo)
        mid.setSizes([420, 420, 420])
        lay.addWidget(mid, 1)

        # 统计胶囊（汇总）+ 进度
        box, bl = _card("运行状态")
        chips = QHBoxLayout()
        chips.setSpacing(8)
        self._class_keys = list(CLASS_COLORS.keys())
        self.chips_cls = {}
        for name, color in CLASS_COLORS.items():
            c = StatChip(CLASS_LABELS.get(name, name), color)
            self.chips_cls[name] = c
            chips.addWidget(c)
        self.chip_total = StatChip("检测总数", "#0E9488")
        chips.addWidget(self.chip_total)
        chips.addStretch(1)
        self.lbl_time = QLabel("耗时 0.0s")
        self.lbl_time.setObjectName("hint")
        chips.addWidget(self.lbl_time)
        bl.addLayout(chips)

        prow = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFixedHeight(12)
        prow.addWidget(self.progress_bar, 1)
        self.lbl_progress = QLabel("0 / 0")
        self.lbl_progress.setObjectName("hint")
        self.lbl_progress.setFixedWidth(90)
        self.lbl_progress.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        prow.addWidget(self.lbl_progress)
        bl.addLayout(prow)

        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setFixedHeight(118)
        bl.addWidget(self.log_box)
        lay.addWidget(box)
        return area

    # ================= 信号 =================
    def _connect_signals(self):
        self.btn_input_browse.clicked.connect(self._on_browse_input)
        self.btn_input_clear.clicked.connect(self._on_clear_input)
        self.btn_output_browse.clicked.connect(self._on_browse_output)
        self.btn_output_open.clicked.connect(self._on_open_output)
        self.btn_start.clicked.connect(self._on_start)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_clear_log.clicked.connect(self.log_box.clear)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.recursive_check.stateChanged.connect(lambda _: self._refresh_input_info())
        self.output_path_edit.textChanged.connect(self._refresh_output_state)

    # ================= 设置记忆 =================
    def _restore_settings(self):
        out = self.settings.value("output_dir", "", type=str)
        if not out:
            out = default_output_dir()
        self.output_path_edit.setText(out)

        last_in = self.settings.value("last_input_dir", "", type=str)
        self._last_input_dir = last_in if os.path.isdir(last_in) else ""

        conf = self.settings.value("conf", 0.25, type=float)
        iou = self.settings.value("iou", 0.45, type=float)
        self.conf_spin.setValue(conf)
        self.iou_spin.setValue(iou)

        self._on_mode_changed()
        self._refresh_output_state()

    def _save_settings(self):
        self.settings.setValue("output_dir", self.output_path_edit.text().strip())
        self.settings.setValue("last_input_dir", getattr(self, "_last_input_dir", ""))
        self.settings.setValue("conf", self.conf_spin.value())
        self.settings.setValue("iou", self.iou_spin.value())

    # ================= 输入 =================
    def _on_mode_changed(self):
        batch = self.mode_combo.currentIndex() == 1
        self.recursive_check.setVisible(batch)
        self.btn_input_browse.setText("选择文件夹…" if batch else "选择图片…")
        self.input_path_edit.clear()
        self.input_info.setText("")
        self._inputs_cache = []

    def _on_browse_input(self):
        start = getattr(self, "_last_input_dir", "") or ""
        if self.mode_combo.currentIndex() == 0:
            p, _ = QFileDialog.getOpenFileName(
                self, "选择黄鳝图像", start,
                "图片 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp);;所有文件 (*)"
            )
        else:
            # 用非原生对话框，这样用户可以在里面「新建文件夹」
            dlg = QFileDialog(self, "选择图片文件夹", start)
            dlg.setFileMode(QFileDialog.Directory)
            dlg.setOption(QFileDialog.ShowDirsOnly, True)
            dlg.setOption(QFileDialog.DontUseNativeDialog, True)
            if dlg.exec_() == QDialog.Accepted:
                p = dlg.selectedFiles()[0] if dlg.selectedFiles() else ""
            else:
                p = ""
        if p:
            self.input_path_edit.setText(p)
            self._last_input_dir = p if os.path.isdir(p) else os.path.dirname(p)
            self._refresh_input_info()
            self._save_settings()

    def _on_clear_input(self):
        self.input_path_edit.clear()
        self.input_info.setText("")
        self._inputs_cache = []

    def _collect_inputs(self):
        p = self.input_path_edit.text().strip()
        if not p:
            return None, "请先选择输入图片或文件夹"
        if not os.path.exists(p):
            return None, f"路径不存在：{p}"
        if os.path.isdir(p):
            files = Pipeline.list_images(p, recursive=self.recursive_check.isChecked())
            if not files:
                return None, "该文件夹下没有找到支持的图片（jpg/png/bmp/tif/webp）"
            return files, ""
        return [p], ""

    def _refresh_input_info(self):
        p = self.input_path_edit.text().strip()
        if not p or not os.path.exists(p):
            self.input_info.setText("")
            self._inputs_cache = []
            return
        if os.path.isdir(p):
            files = Pipeline.list_images(p, recursive=self.recursive_check.isChecked())
            self._inputs_cache = files
            self.input_info.setText(
                f"共发现 {len(files)} 张图片" if files else "⚠ 该文件夹下没有支持的图片"
            )
        else:
            self._inputs_cache = [p]
            self.input_info.setText("单张图片")

    # ================= 输出 =================
    def _on_browse_output(self):
        start = self.output_path_edit.text().strip() or default_output_dir()
        dlg = QFileDialog(self, "选择输出目录（可点『新建文件夹』）", start)
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        if dlg.exec_() == QDialog.Accepted and dlg.selectedFiles():
            self.output_path_edit.setText(dlg.selectedFiles()[0])
            self._save_settings()

    def _on_open_output(self):
        p = self.output_path_edit.text().strip()
        if not p or not os.path.isdir(p):
            QMessageBox.information(self, "提示", "输出目录还不存在，先跑一次或手动创建。")
            return
        try:
            os.startfile(p) if sys.platform == "win32" else None
        except Exception as e:
            QMessageBox.warning(self, "打不开", str(e))

    def _refresh_output_state(self):
        p = self.output_path_edit.text().strip()
        if not p:
            self.output_state.setText("⚠ 未指定输出目录，结果只预览不保存")
            self.output_state.setStyleSheet("color:#D98324; font-size:11px;")
            return
        ok, msg = ensure_output_dir(p)
        if ok:
            self.output_state.setText(f"✅ 可写：{os.path.abspath(p)}")
            self.output_state.setStyleSheet("color:#3FAE5A; font-size:11px;")
        else:
            self.output_state.setText(f"❌ {msg}")
            self.output_state.setStyleSheet("color:#D9534F; font-size:11px;")

    # ================= 启动 =================
    def _on_start(self):
        inputs, err = self._collect_inputs()
        if err:
            QMessageBox.warning(self, "输入有问题", err)
            return

        out = self.output_path_edit.text().strip()
        if out:
            ok, msg = ensure_output_dir(out)
            if not ok:
                QMessageBox.warning(
                    self, "输出目录不可用",
                    f"无法写入：\n{out}\n\n原因：{msg}\n\n"
                    f"请重新选择一个你有权限写入的目录（可在选择框里点『新建文件夹』）。"
                )
                return
        else:
            ret = QMessageBox.question(
                self, "未指定输出目录",
                "没有设置输出目录，结果图将不会保存到磁盘（只在界面预览）。\n仍要继续吗？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if ret != QMessageBox.Yes:
                return

        self._save_settings()

        self.panel_orig.clear()
        self.panel_mask.clear()
        self.panel_yolo.clear()
        self.log_box.clear()
        self.progress_bar.setValue(0)
        self._reset_stats()

        self.worker = InferenceWorker(
            inputs=inputs,
            save_dir=out if out else None,
            conf_thres=self.conf_spin.value(),
            iou_thres=self.iou_spin.value(),
            pipeline=self.pipeline,
        )
        self.worker.progress.connect(self._on_progress)
        self.worker.log.connect(self._append_log)
        self.worker.finished_ok.connect(self._on_finished)
        self.worker.error.connect(self._on_error)
        self.worker.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self._t0 = time.time()
        self.statusBar().showMessage(f"推理中，共 {len(inputs)} 张图…")
        self._append_log(f"[{datetime.now():%H:%M:%S}] 输出目录：{out or '（不保存）'}")

    def _on_stop(self):
        if self.worker:
            self.worker.stop()
            self._append_log("⚠ 已请求停止，等待当前图完成…")
            self.btn_stop.setEnabled(False)

    # ================= 回调 =================
    def _on_progress(self, done, total, last):
        self.progress_bar.setValue(int(done / max(total, 1) * 100))
        self.lbl_progress.setText(f"{done} / {total}")
        if not isinstance(last, dict):
            return
        if "error" in last:
            self._append_log(
                f"❌ {last.get('source_name', '')} 处理失败：{last.get('error', '')}"
            )
        else:
            self.panel_orig.show_pil(last.get("original"))
            self.panel_mask.show_pil(last.get("mask"))
            self.panel_yolo.show_pil(last.get("annotated"))
            self._set_stats(last.get("detections", []))
            # ★ 每条鱼各部位椎骨数量写进日志
            cnt = count_by_class(last.get("detections", []), self._class_keys)
            dets = last.get("detections", [])
            confs = [d.get("conf", 0.0) for d in dets]
            avg = sum(confs) / len(confs) if confs else 0.0
            detail = " / ".join(f"{k} {cnt.get(k, 0)}" for k in self._class_keys)
            self._append_log(
                f"[{datetime.now():%H:%M:%S}] {last.get('source_name', '')} → "
                f"{detail}，合计 {len(dets)} 个，平均置信度 {avg:.2f}，"
                f"耗时 {last.get('elapsed', 0):.2f}s"
            )
            self.statusBar().showMessage(
                f"已完成 {done}/{total} — 当前：{last.get('source_name')}"
            )
        if done == total:
            self.lbl_time.setText(f"耗时 {time.time() - self._t0:.1f}s")

    def _on_finished(self, results):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.progress_bar.setValue(100)
        self.current_results = results

        st = summarize(results)
        self._set_stats_cumulative(st.get("per_class", {}))
        self.lbl_time.setText(f"耗时 {time.time() - self._t0:.1f}s")

        self._append_log(
            f"✅ 完成！成功 {st['ok']} 张 / 失败 {st['fail']} 张，"
            f"累计检测到 {st['detections']} 个椎骨目标（每条明细见上方日志）"
        )
        self.statusBar().showMessage(
            f"完成：成功 {st['ok']}，失败 {st['fail']}，累计检测 {st['detections']} 个"
        )

        if st["fail"]:
            fails = [r for r in results if "error" in r][:5]
            detail = "\n".join(f"· {r.get('source_name')}: {r.get('error')}" for r in fails)
            more = "" if st["fail"] <= 5 else f"\n…（另有 {st['fail'] - 5} 张，详见日志）"
            QMessageBox.warning(self, "部分图片处理失败",
                                f"{detail}{more}\n\n结果图已保存在输出目录。")

    def _on_error(self, msg):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._append_log(f"❌ 运行错误:\n{msg}")
        QMessageBox.critical(self, "运行错误", msg[:1500])
        self.statusBar().showMessage("出错")

    def _append_log(self, text):
        self.log_box.append(text)
        sb = self.log_box.verticalScrollBar()
        sb.setValue(sb.maximum())

    # ================= 统计 =================
    def _reset_stats(self):
        self.chip_total.set_value(0)
        for c in self.chips_cls.values():
            c.set_value(0)
        self.lbl_time.setText("耗时 0.0s")

    def _set_stats(self, detections):
        """显示当前这张图（这一条鱼）的各类椎骨数量"""
        cnt = count_by_class(detections, self._class_keys)
        for k, c in self.chips_cls.items():
            c.set_value(cnt.get(k, 0))
        self.chip_total.set_value(len(detections or []))

    def _set_stats_cumulative(self, per_class):
        """跑完后显示整批的累计数量"""
        total = 0
        for k, c in self.chips_cls.items():
            v = per_class.get(k, 0)
            total += v
            c.set_value(v)
        self.chip_total.set_value(total)

    # ================= 关闭 =================
    def closeEvent(self, event):
        self._save_settings()
        super().closeEvent(event)


# ---------------- 入口 ----------------
def run_app(pipeline=None):
    """打开主窗口。pipeline 为开屏阶段预加载的模型（可为空）"""
    if sys.platform == "win32":
        try:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))
    app.setStyleSheet(APP_QSS)
    win = MainWindow(pipeline=pipeline)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    run_app()
