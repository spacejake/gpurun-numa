"""Parse and cache ``nvidia-smi topo -m`` GPU → CPU / NUMA affinity."""

from __future__ import annotations

import json
import os
import re
import shutil
import socket
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PinMode = Literal["auto", "numactl", "taskset", "none"]

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_GPU_ID_RE = re.compile(r"^GPU(\d+)$", re.IGNORECASE)
_CPU_RANGE_PART = re.compile(r"^(\d+)(?:-(\d+))?$")

TOPOLOGY_CACHE_VERSION = 1


@dataclass(frozen=True)
class GpuAffinity:
    cpu_range: str
    numa_node: int


@dataclass(frozen=True)
class TopologySnapshot:
    hostname: str
    gpu_count: int
    raw_topo: str
    gpus: dict[int, GpuAffinity]


def topology_cache_dir() -> Path:
    override = os.environ.get("GPURUN_CACHE_DIR")
    if override:
        return Path(override)
    return Path.home() / ".cache" / "gpurun"


def topology_cache_path() -> Path:
    return topology_cache_dir() / "topology.json"


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def parse_gpu_ids(spec: str) -> list[int]:
    ids: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise ValueError(f"invalid GPU id {part!r} in {spec!r}")
        ids.append(int(part))
    if not ids:
        raise ValueError("expected at least one GPU index")
    return ids


def gpu_ids_from_env() -> list[int] | None:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or not raw.strip():
        return None
    return parse_gpu_ids(raw)


def cuda_visible_devices_str(gpu_ids: list[int]) -> str:
    return ",".join(str(g) for g in gpu_ids)


def _split_topo_row(line: str) -> list[str]:
    if "\t" in line:
        cols = [c.strip() for c in line.split("\t")]
    else:
        cols = line.split()
    return [c for c in cols if c]


def normalize_cpu_affinity(raw: str) -> str:
    """Return a taskset/numactl-compatible CPU list."""
    text = raw.strip()
    if not text or text.upper() in ("N/A", "NA", "NONE", "-"):
        raise ValueError(f"CPU affinity not available ({raw!r})")

    parts = [p.strip() for p in re.split(r"[, ]+", text) if p.strip()]
    normalized: list[str] = []
    for part in parts:
        m = _CPU_RANGE_PART.match(part)
        if m is None:
            raise ValueError(f"unrecognized CPU affinity token {part!r} in {raw!r}")
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) is not None else lo
        if hi < lo:
            raise ValueError(f"invalid CPU range {part!r}")
        normalized.append(f"{lo}-{hi}" if hi != lo else str(lo))
    return ",".join(normalized)


def _extract_gpu_row_affinity(cols: list[str]) -> tuple[str, str]:
    tail = list(cols)
    while tail and tail[-1].upper() in ("N/A", "NA", "NONE", "-"):
        tail.pop()
    if len(tail) < 3:
        raise ValueError(f"too few columns in GPU topo row: {cols!r}")
    return tail[-2], tail[-1]


def _find_affinity_columns(cols: list[str]) -> tuple[int | None, int | None]:
    cpu_col: int | None = None
    numa_col: int | None = None
    for i, name in enumerate(cols):
        low = name.lower()
        if "cpu" in low and "affinity" in low:
            cpu_col = i
        elif "numa" in low and "affinity" in low:
            numa_col = i
    return cpu_col, numa_col


def parse_topo_text(text: str) -> dict[int, GpuAffinity]:
    """Parse ``nvidia-smi topo -m`` stdout into per-GPU affinity."""
    text = strip_ansi(text)
    header_seen = False
    affinities: dict[int, GpuAffinity] = {}

    for line in text.splitlines():
        cols = _split_topo_row(line)
        if not cols:
            continue

        row_cpu, row_numa = _find_affinity_columns(cols)
        if row_cpu is not None and row_numa is not None:
            header_seen = True
            continue

        joined = " ".join(cols).lower()
        if not header_seen and "cpu affinity" in joined and "numa affinity" in joined:
            header_seen = True
            continue

        if not header_seen:
            continue

        label = cols[0]
        m = _GPU_ID_RE.match(label)
        if m is None:
            continue

        gpu_idx = int(m.group(1))
        try:
            cpu_raw, numa_raw = _extract_gpu_row_affinity(cols)
        except ValueError:
            continue
        try:
            cpu_range = normalize_cpu_affinity(cpu_raw)
        except ValueError as exc:
            raise RuntimeError(
                f"could not parse CPU affinity for {label}: {cpu_raw!r} ({exc})"
            ) from exc
        if not numa_raw.isdigit():
            raise RuntimeError(
                f"could not parse NUMA node for {label}: {numa_raw!r}"
            )

        affinities[gpu_idx] = GpuAffinity(cpu_range=cpu_range, numa_node=int(numa_raw))

    if not affinities:
        raise RuntimeError(
            "no GPU rows found in nvidia-smi topo -m output "
            "(driver may not report CPU affinity on this machine)"
        )
    return affinities


def _host_fingerprint() -> dict[str, str | int]:
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("nvidia-smi not found")
    proc = subprocess.run(
        ["nvidia-smi", "-L"],
        check=True,
        capture_output=True,
        text=True,
    )
    gpu_count = sum(
        1 for line in proc.stdout.splitlines() if line.strip().startswith("GPU ")
    )
    if gpu_count == 0:
        raise RuntimeError("nvidia-smi -L reported no GPUs")
    return {"hostname": socket.gethostname(), "gpu_count": gpu_count}


def _snapshot_to_cache_dict(snap: TopologySnapshot) -> dict[str, Any]:
    return {
        "version": TOPOLOGY_CACHE_VERSION,
        "hostname": snap.hostname,
        "gpu_count": snap.gpu_count,
        "raw_topo": snap.raw_topo,
        "gpus": {
            str(k): {"cpu_range": v.cpu_range, "numa_node": v.numa_node}
            for k, v in sorted(snap.gpus.items())
        },
    }


def _snapshot_from_cache_dict(data: dict[str, Any]) -> TopologySnapshot:
    gpus = {
        int(k): GpuAffinity(
            cpu_range=str(v["cpu_range"]),
            numa_node=int(v["numa_node"]),
        )
        for k, v in data["gpus"].items()
    }
    return TopologySnapshot(
        hostname=str(data["hostname"]),
        gpu_count=int(data["gpu_count"]),
        raw_topo=str(data["raw_topo"]),
        gpus=gpus,
    )


def _cache_matches(data: dict[str, Any], fp: dict[str, str | int]) -> bool:
    return (
        data.get("version") == TOPOLOGY_CACHE_VERSION
        and data.get("hostname") == fp["hostname"]
        and data.get("gpu_count") == fp["gpu_count"]
        and isinstance(data.get("gpus"), dict)
        and isinstance(data.get("raw_topo"), str)
    )


def read_topology_cache() -> dict[str, Any] | None:
    path = topology_cache_path()
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def write_topology_cache(snap: TopologySnapshot) -> Path:
    path = topology_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _snapshot_to_cache_dict(snap)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _fetch_topo_from_nvidia() -> tuple[str, dict[int, GpuAffinity]]:
    proc = subprocess.run(
        ["nvidia-smi", "topo", "-m"],
        check=True,
        capture_output=True,
        text=True,
    )
    raw = strip_ansi(proc.stdout)
    return raw, parse_topo_text(raw)


def get_topology_snapshot(
    *,
    refresh: bool = False,
    use_cache: bool = True,
) -> TopologySnapshot:
    """Load topology; cache at ``~/.cache/gpurun/topology.json`` when possible."""
    fp: dict[str, str | int] | None = None
    if use_cache:
        fp = _host_fingerprint()
        if not refresh:
            cached = read_topology_cache()
            if cached is not None and _cache_matches(cached, fp):
                return _snapshot_from_cache_dict(cached)

    raw, gpus = _fetch_topo_from_nvidia()
    if fp is None:
        fp = {"hostname": socket.gethostname(), "gpu_count": len(gpus)}
    snap = TopologySnapshot(
        hostname=str(fp["hostname"]),
        gpu_count=int(fp["gpu_count"]),
        raw_topo=raw,
        gpus=gpus,
    )
    if use_cache:
        write_topology_cache(snap)
    return snap


def affinity_for_gpus(
    gpu_ids: list[int],
    *,
    refresh: bool = False,
    use_cache: bool = True,
) -> tuple[str, list[int]]:
    """Return ``(cpu_ranges, numa_nodes)`` for the given physical GPU indices."""
    table = get_topology_snapshot(refresh=refresh, use_cache=use_cache).gpus
    missing = [g for g in gpu_ids if g not in table]
    if missing:
        known = ", ".join(str(g) for g in sorted(table))
        raise RuntimeError(
            f"no CPU affinity for GPU(s) {missing}; known GPUs: {known or '(none)'}"
        )

    cpu_ranges: list[str] = []
    numa_nodes: list[int] = []
    seen_cpus: set[str] = set()
    seen_numa: set[int] = set()
    for gpu in gpu_ids:
        aff = table[gpu]
        if aff.cpu_range not in seen_cpus:
            cpu_ranges.append(aff.cpu_range)
            seen_cpus.add(aff.cpu_range)
        if aff.numa_node not in seen_numa:
            numa_nodes.append(aff.numa_node)
            seen_numa.add(aff.numa_node)

    return ",".join(cpu_ranges), numa_nodes


def format_topology_table(snap: TopologySnapshot) -> str:
    lines = ["GPU  CPUs (physcpubind)     NUMA"]
    for idx in sorted(snap.gpus):
        aff = snap.gpus[idx]
        lines.append(f"{idx:3d}  {aff.cpu_range:22s}  {aff.numa_node}")
    return "\n".join(lines)
