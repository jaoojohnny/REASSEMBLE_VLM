"""記錄執行環境（Python、套件、記憶體、磁碟、GPU），方便核對規格中的硬體條件。"""

from __future__ import annotations

import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .util import human_bytes, now_iso

PACKAGES = ["h5py", "numpy", "yaml", "requests", "tqdm", "av", "cv2", "pytest"]


def _pkg_versions() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in PACKAGES:
        try:
            m = importlib.import_module(name)
            out[name] = getattr(m, "__version__", "installed")
            if name == "h5py":
                out["hdf5"] = getattr(getattr(m, "version", None), "hdf5_version", "?")
        except Exception as e:  # noqa: BLE001
            out[name] = f"MISSING ({type(e).__name__})"
    return out


def _total_ram() -> int | None:
    try:
        if os.name == "nt":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))  # type: ignore[attr-defined]
            return int(stat.ullTotalPhys)
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except Exception:  # noqa: BLE001
        return None
    return None


def _gpus() -> dict[str, Any]:
    exe = shutil.which("nvidia-smi")
    if not exe and os.name == "nt":
        cand = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "nvidia-smi.exe"
        exe = str(cand) if cand.exists() else None
    if not exe:
        return {"available": False, "note": "找不到 nvidia-smi（沒有 NVIDIA GPU 或驅動未安裝）"}
    try:
        res = subprocess.run(
            [exe, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=20, check=False,
        )
        gpus = []
        for line in res.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({"name": parts[0], "memory_total": parts[1], "driver": parts[2]})
        return {"available": bool(gpus), "gpus": gpus, "stderr": res.stderr.strip()[:300] or None}
    except Exception as e:  # noqa: BLE001
        return {"available": False, "note": f"nvidia-smi 執行失敗：{e}"}


def collect() -> dict[str, Any]:
    du = shutil.disk_usage(Path.cwd())
    return {
        "collected_at": now_iso(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "cwd": str(Path.cwd()),
        "cpu_count": os.cpu_count(),
        "ram_total": human_bytes(_total_ram()),
        "disk_free": human_bytes(du.free),
        "packages": _pkg_versions(),
        "gpu": _gpus(),
    }


def format_text(info: dict[str, Any]) -> str:
    lines = [
        f"collected_at : {info['collected_at']}",
        f"platform     : {info['platform']} ({info['machine']})",
        f"python       : {info['python']}  ({info['python_executable']})",
        f"conda_env    : {info['conda_env']}",
        f"cpu / ram    : {info['cpu_count']} cores / {info['ram_total']}",
        f"disk free    : {info['disk_free']}  (cwd {info['cwd']})",
        "packages     : " + ", ".join(f"{k}={v}" for k, v in info["packages"].items()),
    ]
    g = info["gpu"]
    if g.get("available"):
        for x in g["gpus"]:
            lines.append(f"gpu          : {x['name']} | {x['memory_total']} | driver {x['driver']}")
    else:
        lines.append(f"gpu          : none ({g.get('note')})")
    return "\n".join(lines)


def write(out_dir: str | Path) -> dict[str, Any]:
    info = collect()
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "envinfo.json").write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    (d / "envinfo.txt").write_text(format_text(info) + "\n", encoding="utf-8")
    return info
