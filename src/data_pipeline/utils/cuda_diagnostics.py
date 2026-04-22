"""CUDA diagnostics helpers.

This repo runs on mixed environments (login nodes, compute nodes, containers).
Sometimes `nvidia-smi` works but CUDA driver initialization fails for user
processes (common root causes: missing device permissions, scheduler not
granting GPU access, misconfigured /dev/nvidia-caps permissions).
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import os

import torch


@dataclass(frozen=True)
class CudaDiagnostics:
    cuinit_rc: int
    cuinit_error: str
    device_count_rc: int
    device_count_error: str
    device_count: int
    can_open_nvidia0: bool
    can_open_nvidiactl: bool
    can_open_uvm: bool
    can_open_cap1: bool
    can_open_cap2: bool


def _can_open(path: str) -> bool:
    try:
        fd = os.open(path, os.O_RDONLY)
        os.close(fd)
        return True
    except Exception:
        return False


def _cuda_errstr(lib, code: int) -> str:
    try:
        cuGetErrorString = lib.cuGetErrorString
        cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        cuGetErrorString.restype = ctypes.c_int
        s = ctypes.c_char_p()
        r = cuGetErrorString(int(code), ctypes.byref(s))
        if r != 0 or not s.value:
            return f"<no errstr r={r}>"
        return s.value.decode("utf-8", "replace")
    except Exception:
        return "<unknown>"


def diagnose_cuda_driver() -> CudaDiagnostics:
    """
    Diagnose CUDA driver API initialization using libcuda directly.

    This avoids torch and gives clearer signals when CUDA is blocked by OS/device
    permissions.
    """
    can_open_nvidia0 = _can_open("/dev/nvidia0")
    can_open_nvidiactl = _can_open("/dev/nvidiactl")
    can_open_uvm = _can_open("/dev/nvidia-uvm")
    can_open_cap1 = _can_open("/dev/nvidia-caps/nvidia-cap1")
    can_open_cap2 = _can_open("/dev/nvidia-caps/nvidia-cap2")

    lib = ctypes.CDLL("libcuda.so.1")
    cuInit = lib.cuInit
    cuInit.argtypes = [ctypes.c_uint]
    cuInit.restype = ctypes.c_int

    cuDeviceGetCount = lib.cuDeviceGetCount
    cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    cuDeviceGetCount.restype = ctypes.c_int

    cuinit_rc = int(cuInit(0))
    cuinit_err = _cuda_errstr(lib, cuinit_rc)

    count = ctypes.c_int(-1)
    device_count_rc = int(cuDeviceGetCount(ctypes.byref(count)))
    device_count_err = _cuda_errstr(lib, device_count_rc)

    return CudaDiagnostics(
        cuinit_rc=cuinit_rc,
        cuinit_error=cuinit_err,
        device_count_rc=device_count_rc,
        device_count_error=device_count_err,
        device_count=int(count.value),
        can_open_nvidia0=can_open_nvidia0,
        can_open_nvidiactl=can_open_nvidiactl,
        can_open_uvm=can_open_uvm,
        can_open_cap1=can_open_cap1,
        can_open_cap2=can_open_cap2,
    )


def resolve_device(device_preference: str | None = "cuda", *, require_cuda: bool = False) -> str:
    """Resolve a requested device preference to a concrete runtime device string."""
    pref = str(device_preference or "cuda").strip().lower()
    if pref.startswith("cpu"):
        return "cpu"
    if pref.startswith("cuda"):
        if torch.cuda.is_available():
            return "cuda"
        if require_cuda:
            require_cuda_or_raise()
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def require_cuda_or_raise() -> None:
    diag = diagnose_cuda_driver()
    if diag.cuinit_rc == 0 and diag.device_count > 0:
        return

    msg = (
        "CUDA requested but CUDA driver initialization failed.\n"
        f"cuInit rc={diag.cuinit_rc} ({diag.cuinit_error})\n"
        f"cuDeviceGetCount rc={diag.device_count_rc} ({diag.device_count_error}) count={diag.device_count}\n"
        "Device-node access:\n"
        f"  /dev/nvidia0: {diag.can_open_nvidia0}\n"
        f"  /dev/nvidiactl: {diag.can_open_nvidiactl}\n"
        f"  /dev/nvidia-uvm: {diag.can_open_uvm}\n"
        f"  /dev/nvidia-caps/nvidia-cap1: {diag.can_open_cap1}\n"
        f"  /dev/nvidia-caps/nvidia-cap2: {diag.can_open_cap2}\n"
        "\n"
        "Most common causes:\n"
        "  - Running on a node where GPUs are visible but compute access is blocked unless you are inside a scheduler allocation.\n"
        "  - Misconfigured /dev/nvidia-caps permissions (cap1 often needs to be group-accessible).\n"
    )
    raise RuntimeError(msg)

