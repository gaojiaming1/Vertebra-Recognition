# -*- coding: utf-8 -*-
"""
打包依赖完整性检查（防止「本机能跑、别人电脑崩」）。

背景：
  曾经为了瘦身，把 torch 的 cusolver / cufft / cusparse / cudnn 训练库等 CUDA DLL
  从 exe 里排除了。本机测试完全正常——但这台机器装了 CUDA Toolkit，系统 PATH 里
  本来就有这些库，所以"测试通过"是假象。换到没装 CUDA 的电脑：
      OSError: [WinError 126] 加载 torch\\lib\\shm.dll 或其依赖失败
  因此必须有**静态的**依赖检查，而不是靠试跑。

本脚本做的事：
  1. 读 PyInstaller 生成的 PKG/EXE TOC，得到「包内有哪些 DLL」
  2. 逐个解析 PE 导入表，拿到它依赖的 DLL 名字
  3. 判断每个被依赖的 DLL：
       - 在包内？             → OK
       - 在系统目录能找到？   → OK（系统库不需要打包）
       - 否则                → ❌ 缺失（这就是别人电脑崩溃的元凶）

用法：
    python tools/check_deps.py [build目录]      # 默认 build_v6/unet_yolo_pipeline
"""
import os
import struct
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 这些是 Windows 系统库，本来就存在于任何 Windows 上，不需要打包
SYSTEM_DIRS = [
    os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32"),
    os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "SysWOW64"),
    os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "WinSxS"),
]


def pe_imports(path):
    """解析 PE 文件的导入表，返回它依赖的 DLL 名列表（小写）"""
    try:
        with open(path, "rb") as f:
            data = f.read()
    except Exception:
        return None

    if len(data) < 0x40 or data[:2] != b"MZ":
        return None
    try:
        e_lfanew = struct.unpack_from("<I", data, 0x3C)[0]
        if data[e_lfanew:e_lfanew + 4] != b"PE\0\0":
            return None
        coff = e_lfanew + 4
        machine, = struct.unpack_from("<H", data, coff)
        opt_off = coff + 20
        magic, = struct.unpack_from("<H", data, opt_off)
        if magic == 0x10B:        # PE32
            dir_off = opt_off + 96
        elif magic == 0x20B:      # PE32+
            dir_off = opt_off + 112
        else:
            return None
        # 数据目录第 1 项 = Import Table (RVA, size)
        imp_rva, imp_size = struct.unpack_from("<II", data, dir_off + 8)
        if not imp_rva:
            return []

        # 节表 → RVA 到文件偏移的映射
        nsec, = struct.unpack_from("<H", data, coff + 2)
        opt_size, = struct.unpack_from("<H", data, coff + 16)
        sec_off = opt_off + opt_size
        sections = []
        for i in range(nsec):
            so = sec_off + i * 40
            vsize, vaddr, rsize, raddr = struct.unpack_from("<IIII", data, so + 8)
            if rsize == 0:      # 未初始化数据用 vsize
                rsize = max(vsize, 1)
            sections.append((vaddr, rsize, raddr))

        def rva2off(rva):
            for vaddr, rsize, raddr in sections:
                if vaddr <= rva < vaddr + rsize:
                    return raddr + (rva - vaddr)
            return None

        off = rva2off(imp_rva)
        if off is None:
            return None

        names = []
        pos = off
        for _ in range(512):            # 描述符数量上限
            chunk = data[pos:pos + 20]
            if len(chunk) < 20:
                break
            (orig_thunk, _, _, name_rva, first_thunk) = struct.unpack_from("<IIIII", chunk, 0)
            if not name_rva and not first_thunk:
                break
            if name_rva:
                noff = rva2off(name_rva)
                if noff is not None:
                    end = data.find(b"\0", noff)
                    if end > noff:
                        nm = data[noff:end].decode("ascii", errors="ignore").strip()
                        if nm:
                            names.append(nm.lower())
            pos += 20
        return names
    except Exception:
        return None


def find_toc(build_dir):
    for name in ("PKG-00.toc", "EXE-00.toc"):
        p = os.path.join(build_dir, name)
        if os.path.exists(p):
            return p
    return None


def main():
    build_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "build_v6", "unet_yolo_pipeline")
    toc = find_toc(build_dir)
    if not toc:
        print("找不到 TOC：%s" % build_dir)
        return 1
    print("TOC：%s" % toc)

    import ast
    entries = ast.literal_eval(open(toc, encoding="utf-8", errors="ignore").read())
    if isinstance(entries, tuple):
        entries = entries[2]

    bundled = {}       # basename(lower) -> 源路径
    for name, path, kind in entries:
        if kind in ("BINARY", "EXTENSION", "DATA") and path and os.path.exists(path):
            low = os.path.basename(name).lower()
            if low.endswith((".dll", ".pyd", ".exe")):
                bundled.setdefault(low, (name, path))

    print("包内 DLL/PYD 数量：%d" % len(bundled))
    print("正在解析导入表…（约需十几秒）\n")

    missing = {}
    checked = 0
    for low, (name, path) in bundled.items():
        deps = pe_imports(path)
        if not deps:
            continue
        checked += 1
        for d in deps:
            if d in bundled:
                continue
            # 系统库？
            found = False
            for sd in SYSTEM_DIRS:
                if os.path.exists(os.path.join(sd, d)):
                    found = True
                    break
            if not found:
                missing.setdefault(d, []).append(name)

    print("已检查：%d 个文件" % checked)
    if not missing:
        print("\n✅ 依赖检查通过：包内所有 DLL 的依赖都能在包内或系统目录找到")
        return 0

    print("\n❌ 发现 %d 个缺失依赖（在没装 CUDA/VC 运行库的电脑上会直接崩）：\n" % len(missing))
    for d, users in sorted(missing.items(), key=lambda kv: -len(kv[1]))[:40]:
        print("  %-34s 被 %d 个文件依赖，例如：%s" % (d, len(users), users[0]))
    return 2


if __name__ == "__main__":
    sys.exit(main())
