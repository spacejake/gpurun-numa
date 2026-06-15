"""CLI helpers and launch preview."""

from __future__ import annotations

import io
import socket
import textwrap
from pathlib import Path
from unittest.mock import patch

from gpurun.cli import format_launch_preview, main, print_launch_preview, wants_launch_preview
from gpurun.launcher import build_launch_cmd
from gpurun.topology import TopologySnapshot, parse_topo_text, write_topology_cache


def test_format_launch_preview_quotes_args() -> None:
    launch = ["numactl", "--membind=0", "python", "train.py", "--config", "a b.yaml"]
    text = format_launch_preview(launch, gpu_ids=[4, 5])
    assert text.startswith("export CUDA_VISIBLE_DEVICES=4,5\n")
    assert "'a b.yaml'" in text or '"a b.yaml"' in text


def test_print_launch_preview_to_stream() -> None:
    buf = io.StringIO()
    print_launch_preview(["python", "x.py"], [0], stream=buf)
    assert "export CUDA_VISIBLE_DEVICES=0" in buf.getvalue()
    assert "python x.py" in buf.getvalue()


def test_wants_launch_preview() -> None:
    class Args:
        verbose = False
        show_command = False

    a = Args()
    assert not wants_launch_preview(a)  # type: ignore[arg-type]
    a.verbose = True
    assert wants_launch_preview(a)  # type: ignore[arg-type]
    a.verbose = False
    a.show_command = True
    assert wants_launch_preview(a)  # type: ignore[arg-type]


def _seed_topology_cache(tmp_path: Path) -> str:
    topo = textwrap.dedent(
        """\
        GPU0    CPU Affinity    NUMA Affinity
        GPU0    0-11            0
        """
    )
    host = socket.gethostname()
    write_topology_cache(
        TopologySnapshot(
            hostname=host,
            gpu_count=1,
            raw_topo=topo,
            gpus=parse_topo_text(topo),
        )
    )
    return host


def test_main_dry_run(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("GPURUN_CACHE_DIR", str(tmp_path))
    host = _seed_topology_cache(tmp_path)
    fp = {"hostname": host, "gpu_count": 1}
    with (
        patch("gpurun.topology._host_fingerprint", return_value=fp),
        patch("gpurun.launcher.shutil.which") as which,
    ):
        which.side_effect = lambda c: "/usr/bin/numactl" if c == "numactl" else None
        rc = main(["-g", "0", "--dry-run", "python", "-c", "pass"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "export CUDA_VISIBLE_DEVICES=0" in out
    assert "numactl" in out
    assert "python -c pass" in out


def test_main_show_command_prints_then_exec(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("GPURUN_CACHE_DIR", str(tmp_path))
    host = _seed_topology_cache(tmp_path)
    fp = {"hostname": host, "gpu_count": 1}
    with (
        patch("gpurun.topology._host_fingerprint", return_value=fp),
        patch("gpurun.launcher.shutil.which") as which,
        patch("gpurun.cli.os.execvpe") as execvpe,
    ):
        which.side_effect = lambda c: "/usr/bin/numactl" if c == "numactl" else None
        rc = main(["-g", "0", "--show-command", "python", "x.py"])
    assert rc == 0
    err = capsys.readouterr().err
    assert "export CUDA_VISIBLE_DEVICES=0" in err
    assert "numactl" in err
    execvpe.assert_called_once()
    launch, env = execvpe.call_args[0][1], execvpe.call_args[0][2]
    assert launch[0] == "numactl"
    assert env["CUDA_VISIBLE_DEVICES"] == "0"


def test_main_verbose_prints_launch_preview(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("GPURUN_CACHE_DIR", str(tmp_path))
    host = _seed_topology_cache(tmp_path)
    fp = {"hostname": host, "gpu_count": 1}
    with (
        patch("gpurun.topology._host_fingerprint", return_value=fp),
        patch("gpurun.launcher.shutil.which") as which,
        patch("gpurun.cli.os.execvpe"),
    ):
        which.side_effect = lambda c: "/usr/bin/numactl" if c == "numactl" else None
        rc = main(["-g", "0", "--verbose", "python", "x.py"])
    assert rc == 0
    assert "export CUDA_VISIBLE_DEVICES=0" in capsys.readouterr().err


def test_build_launch_cmd_numactl_preview() -> None:
    """Keep launcher test coverage; preview uses same argv."""
    import textwrap
    from unittest.mock import patch

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
        run.return_value = __import__("subprocess").CompletedProcess(
            args=[], returncode=0, stdout=topo, stderr=""
        )
        which.side_effect = lambda c: "/usr/bin/numactl" if c == "numactl" else None
        launch, _ = build_launch_cmd(
            ["python", "x.py"], gpu_ids=[0], mode="numactl", use_cache=False
        )
    preview = format_launch_preview(launch, [0])
    assert "--physcpubind=0-11,24-35" in preview
    assert "--membind=0" in preview
