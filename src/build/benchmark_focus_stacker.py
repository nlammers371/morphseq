from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

from .export_utils import LoG_focus_stacker


@dataclass
class BenchmarkResult:
    device: str
    batch_size: int
    iterations: int
    total_seconds: float

    @property
    def sec_per_iter(self) -> float:
        return self.total_seconds / max(self.iterations, 1)


def _sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def _run_once(
    data: torch.Tensor,
    filter_size: int,
    device: str,
) -> None:
    LoG_focus_stacker(data.to(device, non_blocking=True), filter_size, device)


def benchmark_device(
    template: torch.Tensor,
    filter_size: int,
    device: str,
    warmup: int,
    iterations: int,
) -> BenchmarkResult:
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU detected.")
        torch.cuda.init()

    data = template.to(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    target_device = str(data.device)

    for _ in range(max(warmup, 0)):
        _run_once(data, filter_size, target_device)
        _sync(target_device)

    start = time.perf_counter()
    for _ in range(iterations):
        _run_once(data, filter_size, target_device)
        _sync(target_device)
    total = time.perf_counter() - start
    return BenchmarkResult(target_device, data.shape[0], iterations, total)


def build_template_tensor(
    batch: int,
    z: int,
    height: int,
    width: int,
    seed: int | None = None,
    from_npy: str | None = None,
) -> torch.Tensor:
    if from_npy:
        arr = np.load(from_npy)
        tensor = torch.from_numpy(arr).float()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4:
            raise ValueError("Loaded stack must be 3D or 4D (Z,Y,X) or (N,Z,Y,X).")
        return tensor

    rng = torch.Generator().manual_seed(seed if seed is not None else 0)
    return torch.rand((batch, z, height, width), generator=rng)


def format_results(results: Sequence[BenchmarkResult]) -> str:
    lines = ["device,batch,iterations,total_s,sec_per_iter"]
    for r in results:
        lines.append(f"{r.device},{r.batch_size},{r.iterations},{r.total_seconds:.3f},{r.sec_per_iter:.3f}")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark LoG focus stacker on CPU vs GPU.")
    parser.add_argument("--batch", type=int, default=4, help="Number of wells per batch.")
    parser.add_argument("--z", type=int, default=17, help="Z slices per stack.")
    parser.add_argument("--height", type=int, default=2189, help="Image height (pixels).")
    parser.add_argument("--width", type=int, default=1152, help="Image width (pixels).")
    parser.add_argument("--filter-size", type=int, default=3, help="LoG filter size.")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations per device.")
    parser.add_argument("--iters", type=int, default=5, help="Timed iterations per device.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for synthetic data.")
    parser.add_argument("--stack-npy", type=str, help="Optional path to .npy tensor (Z,Y,X) or (N,Z,Y,X).")
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cpu", "cuda" if torch.cuda.is_available() else "cpu"],
        help="Devices to benchmark (e.g. cpu cuda:0).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    template = build_template_tensor(
        args.batch,
        args.z,
        args.height,
        args.width,
        args.seed,
        args.stack_npy,
    )

    results: list[BenchmarkResult] = []
    for dev in args.devices:
        try:
            result = benchmark_device(template.clone(), args.filter_size, dev, args.warmup, args.iters)
        except RuntimeError as exc:
            print(f"[WARN] Skipping device {dev}: {exc}")
            continue
        results.append(result)
        print(
            f"{dev}: batch={result.batch_size} iters={result.iterations} "
            f"total={result.total_seconds:.2f}s ({result.sec_per_iter:.2f}s/iter)"
        )

    print("\n" + format_results(results))


if __name__ == "__main__":
    main()
