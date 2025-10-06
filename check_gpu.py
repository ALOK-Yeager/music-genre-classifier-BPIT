"""Utility script to inspect GPU availability for PyTorch workloads.

Run with:
    python check_gpu.py

Optional flags:
    --json              Output results in JSON format for automation.
    --nvidia-smi        Include output from the `nvidia-smi` command if available.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from typing import Any, Dict, List


def safe_import_torch() -> Any:
    """Import torch with a helpful error message if it's missing."""
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - informative error path
        print(
            "PyTorch is not installed or failed to import. "
            "Install dependencies via `pip install -r requirements.txt`.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    return torch


def gather_torch_cuda_info(torch_module: Any) -> Dict[str, Any]:
    """Collect PyTorch CUDA diagnostics without allocating large tensors."""
    info: Dict[str, Any] = {
        "torch_version": getattr(torch_module, "__version__", "unknown"),
        "cuda_available": bool(torch_module.cuda.is_available()),
        "cuda_version": getattr(torch_module.version, "cuda", None),
        "device_count": torch_module.cuda.device_count(),
        "devices": [],
    }

    if not info["cuda_available"]:
        return info

    for index in range(info["device_count"]):
        props = torch_module.cuda.get_device_properties(index)
        device_summary = {
            "index": index,
            "name": props.name,
            "total_memory_gb": round(props.total_memory / (1024**3), 2),
            "multi_processor_count": props.multi_processor_count,
            "capability": f"{props.major}.{props.minor}",
        }
        info["devices"].append(device_summary)

    try:
        current_device = torch_module.cuda.current_device()
    except AssertionError:
        # Raised if no device has been set yet despite availability.
        current_device = 0

    info["active_device_index"] = current_device
    info["active_device_name"] = torch_module.cuda.get_device_name(current_device)

    return info


def maybe_run_nvidia_smi() -> Dict[str, Any]:
    """Run `nvidia-smi` if available to capture GPU utilization."""
    binary = shutil.which("nvidia-smi")
    if not binary:
        return {"available": False, "output": None}

    try:
        completed = subprocess.run(
            [binary, "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv"],
            capture_output=True,
            text=True,
            check=True,
        )
        output = completed.stdout.strip()
    except subprocess.CalledProcessError as exc:  # pragma: no cover - depends on system
        output = exc.stdout or exc.stderr or str(exc)

    return {"available": True, "output": output}


def format_human_readable(info: Dict[str, Any], smi_info: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("🔍 PyTorch GPU Diagnostics")
    lines.append(f"- PyTorch version: {info['torch_version']}")
    lines.append(f"- CUDA toolkit available: {info['cuda_available']}")
    lines.append(f"- CUDA version: {info['cuda_version'] or 'N/A'}")
    lines.append(f"- Detected CUDA devices: {info['device_count']}")

    if info["cuda_available"] and info["device_count"]:
        lines.append("")
        lines.append("GPU Inventory:")
        for device in info["devices"]:
            lines.append(
                f"  • GPU {device['index']}: {device['name']} | "
                f"Compute Capability {device['capability']} | "
                f"Memory: {device['total_memory_gb']} GB | "
                f"SMs: {device['multi_processor_count']}"
            )
        lines.append(
            f"\nActive device index: {info['active_device_index']} "
            f"({info['active_device_name']})"
        )

    if smi_info.get("available"):
        lines.append("")
        lines.append("nvidia-smi output:")
        lines.append(smi_info.get("output") or "<no data>")
    else:
        lines.append("")
        lines.append("nvidia-smi not found on PATH.")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect GPU availability for PyTorch training.")
    parser.add_argument("--json", action="store_true", help="Output diagnostics in JSON format.")
    parser.add_argument(
        "--nvidia-smi",
        action="store_true",
        help="Include output from nvidia-smi if the binary is available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch_module = safe_import_torch()
    cuda_info = gather_torch_cuda_info(torch_module)
    smi_info = maybe_run_nvidia_smi() if args.nvidia_smi else {"available": False, "output": None}

    payload = {"cuda_info": cuda_info, "nvidia_smi": smi_info}

    if args.json:
        print(json.dumps(payload, indent=2))
        return

    print(format_human_readable(cuda_info, smi_info))


if __name__ == "__main__":
    main()
