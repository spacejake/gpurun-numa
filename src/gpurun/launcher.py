"""Build wrapped launch commands with taskset or numactl."""

from __future__ import annotations

import shutil

from gpurun.topology import PinMode, affinity_for_gpus

LaunchResult = tuple[list[str], str | None]


def build_launch_cmd(
    cmd: list[str],
    *,
    gpu_ids: list[int],
    mode: PinMode = "auto",
    refresh_topology: bool = False,
    use_cache: bool = True,
) -> LaunchResult:
    """Wrap *cmd* with CPU/NUMA affinity.

    Returns ``(argv, warning)`` where *warning* is set when mode is ``auto`` and
    no pinning tool is available (caller should use ``--fallback``).
    """
    if mode == "none":
        return list(cmd), None

    cpus, numa_nodes = affinity_for_gpus(
        gpu_ids, refresh=refresh_topology, use_cache=use_cache
    )

    if mode == "numactl":
        if shutil.which("numactl") is None:
            raise RuntimeError("numactl not found")
        return _numactl_cmd(cpus, numa_nodes, cmd), None

    if mode == "taskset":
        if shutil.which("taskset") is None:
            raise RuntimeError("taskset not found")
        return ["taskset", "-c", cpus, *cmd], None

    # auto
    if shutil.which("numactl") is not None:
        return _numactl_cmd(cpus, numa_nodes, cmd), None
    if shutil.which("taskset") is not None:
        return ["taskset", "-c", cpus, *cmd], None
    return list(cmd), (
        "gpurun: numactl and taskset not found; running without CPU pinning"
    )


def _numactl_cmd(cpus: str, numa_nodes: list[int], cmd: list[str]) -> list[str]:
    nodes = ",".join(str(n) for n in numa_nodes)
    return [
        "numactl",
        f"--physcpubind={cpus}",
        f"--membind={nodes}",
        *cmd,
    ]
