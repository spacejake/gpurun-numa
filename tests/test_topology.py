"""Topology parsing and launcher tests."""

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from gpurun.launcher import build_launch_cmd
from gpurun.topology import (
    TopologySnapshot,
    affinity_for_gpus,
    get_topology_snapshot,
    normalize_cpu_affinity,
    parse_topo_text,
    topology_cache_path,
    write_topology_cache,
)


def test_normalize_cpu_affinity_single_and_multi_range() -> None:
    assert normalize_cpu_affinity("0-23") == "0-23"
    assert normalize_cpu_affinity("48-63,64-79") == "48-63,64-79"
    assert normalize_cpu_affinity("0-15 16-31") == "0-15,16-31"


def test_normalize_cpu_affinity_rejects_na() -> None:
    with pytest.raises(ValueError, match="N/A"):
        normalize_cpu_affinity("N/A")


def test_parse_topo_tab_separated() -> None:
    topo = textwrap.dedent(
        """\
        \tGPU0\tGPU1\tCPU Affinity\tNUMA Affinity
        GPU0\t X \t PIX \t0-15\t0
        GPU1\t PIX \t X \t16-31\t0
        """
    )
    table = parse_topo_text(topo)
    assert table[0].cpu_range == "0-15"
    assert table[1].cpu_range == "16-31"
    assert table[0].numa_node == 0


def test_parse_topo_space_separated() -> None:
    topo = textwrap.dedent(
        """\
        GPU0    GPU1    CPU Affinity    NUMA Affinity
        GPU0     X       PIX     32-47           1
        GPU1     PIX     X       48-63           1
        """
    )
    table = parse_topo_text(topo)
    assert table[0].cpu_range == "32-47"
    assert table[1].numa_node == 1


def test_parse_topo_trailing_na_column() -> None:
    topo = textwrap.dedent(
        """\
        GPU0    GPU1    CPU Affinity    NUMA Affinity   GPU NUMA ID
        GPU0     X      NODE    SYS     0-11,24-35      0               N/A
        GPU1    NODE     X      SYS     0-11,24-35      0               N/A
        GPU2    SYS     SYS      X      12-23,36-47     1               N/A
        """
    )
    table = parse_topo_text(topo)
    assert table[0].cpu_range == "0-11,24-35"
    assert table[2].numa_node == 1


def test_affinity_same_numa_dedupes_cpus() -> None:
    topo = textwrap.dedent(
        """\
        GPU0    GPU1    CPU Affinity    NUMA Affinity
        GPU0    12-23,36-47     1
        GPU1    12-23,36-47     1
        """
    )
    with patch("gpurun.topology.subprocess.run") as run:
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=topo, stderr=""
        )
        cpus, numa = affinity_for_gpus([0, 1], use_cache=False)
    assert cpus == "12-23,36-47"
    assert numa == [1]


def test_affinity_different_numa_nodes() -> None:
    topo = textwrap.dedent(
        """\
        GPU0    GPU1    CPU Affinity    NUMA Affinity
        GPU0    0-11,24-35      0
        GPU2    12-23,36-47     1
        """
    )
    with patch("gpurun.topology.subprocess.run") as run:
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=topo, stderr=""
        )
        cpus, numa = affinity_for_gpus([0, 2], use_cache=False)
    assert cpus == "0-11,24-35,12-23,36-47"
    assert numa == [0, 1]


def test_missing_gpu_raises() -> None:
    topo = textwrap.dedent(
        """\
        GPU0    CPU Affinity    NUMA Affinity
        GPU0    0-15            0
        """
    )
    with patch("gpurun.topology.subprocess.run") as run:
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=topo, stderr=""
        )
        with pytest.raises(RuntimeError, match="no CPU affinity"):
            affinity_for_gpus([99], use_cache=False)


def test_build_launch_cmd_numactl() -> None:
    topo = textwrap.dedent(
        """\
        GPU0    CPU Affinity    NUMA Affinity
        GPU0    0-11,24-35      0
        """
    )
    with (
        patch("gpurun.topology.subprocess.run") as run,
        patch("gpurun.launcher.shutil.which") as which,
    ):
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=topo, stderr=""
        )
        which.side_effect = lambda c: "/usr/bin/numactl" if c == "numactl" else None
        launch, warn = build_launch_cmd(
            ["python", "x.py"], gpu_ids=[0], mode="numactl", use_cache=False
        )
    assert warn is None
    assert launch[:4] == [
        "numactl",
        "--physcpubind=0-11,24-35",
        "--membind=0",
        "python",
    ]


def test_build_launch_cmd_auto_taskset_fallback() -> None:
    topo = textwrap.dedent(
        """\
        GPU0    CPU Affinity    NUMA Affinity
        GPU0    32-47           1
        """
    )
    with (
        patch("gpurun.topology.subprocess.run") as run,
        patch("gpurun.launcher.shutil.which") as which,
    ):
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=topo, stderr=""
        )
        which.side_effect = lambda c: "/usr/bin/taskset" if c == "taskset" else None
        launch, warn = build_launch_cmd(
            ["python", "x.py"], gpu_ids=[0], mode="auto", use_cache=False
        )
    assert warn is None
    assert launch == ["taskset", "-c", "32-47", "python", "x.py"]


def test_build_launch_cmd_auto_no_tools_returns_warning() -> None:
    topo = textwrap.dedent(
        """\
        GPU0    CPU Affinity    NUMA Affinity
        GPU0    0-15            0
        """
    )
    with (
        patch("gpurun.topology.subprocess.run") as run,
        patch("gpurun.launcher.shutil.which", return_value=None),
    ):
        run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout=topo, stderr=""
        )
        launch, warn = build_launch_cmd(
            ["python", "x.py"], gpu_ids=[0], mode="auto", use_cache=False
        )
    assert warn is not None
    assert launch == ["python", "x.py"]


def test_topology_cache_roundtrip(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GPURUN_CACHE_DIR", str(tmp_path))
    topo = textwrap.dedent(
        """\
        GPU0    CPU Affinity    NUMA Affinity
        GPU0    0-11,24-35      0
        """
    )
    parsed = parse_topo_text(topo)
    snap = TopologySnapshot(
        hostname="testhost",
        gpu_count=1,
        raw_topo=topo,
        gpus=parsed,
    )
    write_topology_cache(snap)
    assert topology_cache_path() == tmp_path / "topology.json"
    data = json.loads(topology_cache_path().read_text(encoding="utf-8"))
    assert data["gpus"]["0"]["cpu_range"] == "0-11,24-35"


def test_get_topology_snapshot_uses_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("GPURUN_CACHE_DIR", str(tmp_path))
    topo = textwrap.dedent(
        """\
        GPU0    CPU Affinity    NUMA Affinity
        GPU0    32-47           1
        """
    )
    write_topology_cache(
        TopologySnapshot(
            hostname="testhost",
            gpu_count=1,
            raw_topo=topo,
            gpus=parse_topo_text(topo),
        )
    )
    with (
        patch("gpurun.topology.subprocess.run") as run,
        patch(
            "gpurun.topology._host_fingerprint",
            return_value={"hostname": "testhost", "gpu_count": 1},
        ),
    ):
        snap = get_topology_snapshot()
        run.assert_not_called()
    assert snap.gpus[0].cpu_range == "32-47"
