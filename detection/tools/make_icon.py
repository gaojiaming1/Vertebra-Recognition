# -*- coding: utf-8 -*-
"""
生成应用专属图标 app.ico（任务栏 / 资源管理器 / 安装包都用它）。

设计（v2，更现代的「鱼骨 + 检测框」）：
  - 外形：squircle（超椭圆）圆角，比普通圆角矩形更接近现代 App 图标
  - 背景：深潭青 → 亮青 的对角渐变 + 顶部柔光 + 底部暗角
  - 主体：一条完整的**鱼骨**——主脊柱（从头到尾渐细）、上下对称肋骨（中部最长、
          两端渐短，形成经典鱼骨外形）、鱼头圆点、尾鳍三角
  - 点缀：金色 YOLO 检测框四角，框住鱼骨中段
  - 小尺寸（16~40）自动用简化版：更少的肋骨、更粗的线条，保证不糊

用法：  python tools/make_icon.py
产物：  assets/app.ico      （16/20/24/32/40/48/64/128/256 共 9 个尺寸）
        assets/app_512.png （大图，可给桌面快捷方式单独指定）
"""
import math
import os

from PIL import Image, ImageDraw, ImageFilter

try:                       # Pillow ≥ 9.1
    RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:     # Pillow < 9.1
    RESAMPLE = Image.LANCZOS

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(BASE, "assets")

# ---- 配色 ----
BG_TL = (10, 82, 74)        # 左上：深潭青
BG_BR = (26, 197, 168)      # 右下：亮青绿
GLINT = (255, 255, 255)     # 顶部柔光
BONE = (255, 255, 255)      # 鱼骨：纯白
BONE_SOFT = (214, 245, 240)  # 鱼骨次要部分（尾鳍等）
FRAME = (255, 199, 92)      # 检测框：暖金色（青底上的互补色，小尺寸也醒目）
SHADOW = (3, 42, 38)        # 鱼骨投影色

SS = 4                      # supersample：先画大再缩，边缘干净


def _lerp(a, b, t):
    return a + (b - a) * t


def _mix(c1, c2, t):
    return tuple(int(round(_lerp(c1[i], c2[i], t))) for i in range(3))


def _squircle_mask(size, power=4.2):
    """超椭圆（squircle）遮罩：|x/a|^n + |y/a|^n = 1，比圆角矩形更现代"""
    S = size
    m = Image.new("L", (S, S), 0)
    px = m.load()
    half = S / 2.0
    for y in range(S):
        for x in range(S):
            dx = abs(x - half + 0.5) / half
            dy = abs(y - half + 0.5) / half
            if dx ** power + dy ** power <= 1.0:
                # 边缘 1px 抗锯齿
                px[x, y] = 255
    return m


def _background(size):
    """对角渐变 + 顶部柔光 + 底部暗角"""
    img = Image.new("RGB", (size, size))
    px = img.load()
    for y in range(size):
        for x in range(size):
            t = (x * 0.65 + y * 0.35) / size
            px[x, y] = _mix(BG_TL, BG_BR, min(1.0, t))
    img = Image.fromarray(_add_glint(img, size))
    return img.convert("RGBA")


def _add_glint(img, size):
    import numpy as np
    arr = np.asarray(img).astype("float32")
    ys, xs = np.mgrid[0:size, 0:size]
    # 顶部柔光（椭圆衰减）
    cx, cy = size * 0.30, size * 0.16
    d = np.sqrt(((xs - cx) / (size * 0.62)) ** 2 + ((ys - cy) / (size * 0.46)) ** 2)
    glow = np.clip(1.0 - d, 0, 1) ** 2.0 * 0.34
    # 底部暗角
    d2 = np.sqrt(((xs - size * 0.5) / (size * 0.75)) ** 2
                 + ((ys - size * 1.02) / (size * 0.60)) ** 2)
    shade = np.clip(1.0 - d2, 0, 1) ** 2.0 * 0.22
    out = arr * (1.0 + glow[..., None]) * (1.0 - shade[..., None])
    return np.clip(out, 0, 255).astype("uint8")


def _fish_bone(d, box, ribs=9, spine_w=(0.052, 0.018), color=BONE, color2=BONE_SOFT):
    """
    画一条鱼骨：主脊柱 + 上下对称肋骨 + 鱼头 + 尾鳍
    box: (x0, y0, x1, y1) 鱼骨占的区域
    """
    x0, y0, x1, y1 = box
    W = x1 - x0
    H = y1 - y0
    cy = (y0 + y1) / 2.0

    head_x = x0 + W * 0.085
    tail_x = x1 - W * 0.045
    spine_y = cy

    # ---------- 肋骨（先画，压在脊柱下面）----------
    max_len = H * 0.40
    for i in range(ribs):
        t = (i + 0.5) / ribs
        x = _lerp(head_x + W * 0.06, tail_x - W * 0.10, t)
        # 长度：中间最长，两端渐短（sin 包络）
        L = max_len * (math.sin(math.pi * (0.12 + 0.82 * t))) ** 0.85
        # 向尾部倾斜
        skew = W * 0.055
        w_base = max(2.0, W * 0.017)
        tip_up = (x + skew, spine_y - L)
        tip_dn = (x + skew, spine_y + L)
        base_up = (x - w_base, spine_y - w_base * 0.6)
        base_dn = (x + w_base, spine_y - w_base * 0.6)
        d.polygon([base_up, base_dn, tip_up], fill=color)
        base_up2 = (x - w_base, spine_y + w_base * 0.6)
        base_dn2 = (x + w_base, spine_y + w_base * 0.6)
        d.polygon([base_up2, base_dn2, tip_dn], fill=color)

    # ---------- 主脊柱（从头到尾渐细）----------
    steps = 60
    w_head = W * spine_w[0]
    w_tail = W * spine_w[1]
    for i in range(steps):
        t0 = i / steps
        t1 = (i + 1) / steps
        xa = _lerp(head_x, tail_x, t0)
        xb = _lerp(head_x, tail_x, t1) + 0.6
        wa = _lerp(w_head, w_tail, t0)
        wb = _lerp(w_head, w_tail, t1)
        d.polygon([(xa, spine_y - wa), (xb, spine_y - wb),
                   (xb, spine_y + wb), (xa, spine_y + wa)], fill=color)

    # ---------- 鱼头（圆点 + 小眼）----------
    r = W * 0.062
    d.ellipse([head_x - r, spine_y - r, head_x + r, spine_y + r], fill=color)
    er = r * 0.34
    d.ellipse([head_x - r * 0.34 - er, spine_y - er,
               head_x - r * 0.34 + er, spine_y + er], fill=(9, 74, 68))

    # ---------- 尾鳍（三角）----------
    tw = W * 0.075
    th = H * 0.30
    d.polygon([(tail_x - W * 0.012, spine_y),
               (tail_x + tw, spine_y - th),
               (tail_x + tw, spine_y + th)], fill=color2)


def _detect_frame(d, box, lw, corner_ratio=0.26, color=FRAME):
    """YOLO 风格的四角检测框"""
    x1, y1, x2, y2 = box
    corner = min(x2 - x1, y2 - y1) * corner_ratio
    for (sx, sy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                             (x1, y2, 1, -1), (x2, y2, -1, -1)):
        d.line([(sx, sy), (sx + corner * dx, sy)], fill=color, width=lw)
        d.line([(sx, sy), (sx, sy + corner * dy)], fill=color, width=lw)


def build(size=512, simple=False):
    """
    :param simple: 小尺寸简化版（肋骨更少更短、线条更粗、不画鱼眼）
    """
    S = size * SS
    img = _background(S)
    d = ImageDraw.Draw(img, "RGBA")

    if simple:
        ribs = 5
        box = (S * 0.17, S * 0.33, S * 0.86, S * 0.67)
        lw = int(S * 0.036)
    else:
        ribs = 9
        box = (S * 0.16, S * 0.30, S * 0.85, S * 0.70)
        lw = int(S * 0.024)

    # ---- 鱼骨投影（很淡，让主体浮起来）----
    shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    _fish_bone(sd, (box[0], box[1] + S * 0.018, box[2], box[3] + S * 0.018),
               ribs=ribs, color=(0, 30, 28, 120), color2=(0, 30, 28, 90))
    shadow = shadow.filter(ImageFilter.GaussianBlur(S * 0.012))
    img = Image.alpha_composite(img, shadow)
    d = ImageDraw.Draw(img, "RGBA")

    # ---- 鱼骨本体 ----
    _fish_bone(d, box, ribs=ribs)

    # ---- 检测框（框住鱼骨中段）----
    fx1 = box[0] + (box[2] - box[0]) * (0.02 if simple else 0.14)
    fx2 = box[2] - (box[2] - box[0]) * (0.02 if simple else 0.10)
    fy1 = box[1] - S * (0.035 if simple else 0.045)
    fy2 = box[3] + S * (0.035 if simple else 0.045)
    _detect_frame(d, (fx1, fy1, fx2, fy2), lw=lw)

    # ---- squircle 裁切 + 顶部高光描边 ----
    img = img.resize((size, size), RESAMPLE)
    mask = _squircle_mask(size)
    out = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    out.paste(img, (0, 0), mask)

    edge = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ed = ImageDraw.Draw(edge)
    r = int(size * 0.20)
    ed.rounded_rectangle([0, 0, size - 1, size - 1], radius=r,
                         outline=(255, 255, 255, 90), width=max(1, int(size * 0.012)))
    # 只保留 squircle 边缘那一圈（内部描边会被 mask 掉）
    edge.putalpha(Image.composite(edge.split()[3], Image.new("L", (size, size), 0), mask))
    out = Image.alpha_composite(out, edge)
    return out


def ascii_preview(im, w=46):
    """字符画预览（终端里确认轮廓是否正常）"""
    h = int(w * 0.5)
    small = im.convert("RGBA").resize((w, h), RESAMPLE)
    px = small.load()
    ramp = " .:-=+*#%@"
    lines = []
    for y in range(h):
        row = []
        for x in range(w):
            r, g, b, a = px[x, y]
            if a < 40:
                row.append(" ")
            else:
                lum = 0.299 * r + 0.587 * g + 0.114 * b
                row.append(ramp[min(len(ramp) - 1, int(lum / 255 * (len(ramp) - 1)))])
        lines.append("".join(row))
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = build(512)
    base.save(os.path.join(OUT_DIR, "app_512.png"))

    plan = [(16, True), (20, True), (24, True), (32, True), (40, True),
            (48, False), (64, False), (128, False), (256, False)]
    imgs = [build(s, simple=sm) for s, sm in plan]
    base.save(os.path.join(OUT_DIR, "app.ico"),
              sizes=[(s, s) for s, _ in plan], append_images=imgs)

    ico = Image.open(os.path.join(OUT_DIR, "app.ico"))
    print("图标已生成: assets/app.ico")
    print("内含尺寸:", sorted({s[0] for s in ico.info.get("sizes", set())}))
    print()
    print("— 512 完整版 —")
    print(ascii_preview(base))
    print()
    print("— 32 简化版（任务栏小尺寸）—")
    print(ascii_preview(imgs[3], w=40))


if __name__ == "__main__":
    main()
