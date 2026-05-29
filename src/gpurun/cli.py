"""gpurun command-line interface."""

from __future__ import annotations

import argparse
import io
import os
import shlex
import sys

from gpurun.launcher import build_launch_cmd
from gpurun.topology import (
    PinMode,
    cuda_visible_devices_str,
    format_topology_table,
    get_topology_snapshot,
    gpu_ids_from_env,
    parse_gpu_ids,
    topology_cache_path,
)

_PROG = "gpurun"

_EPILOG = """
Examples:
  gpurun -g 4 python train.py --config config.yaml

  gpurun -g 4,5 torchrun --standalone --nproc_per_node=2 train.py --config config.yaml

  gpurun -g 0,1 --dry-run torchrun --standalone --nproc_per_node=2 train.py ...

  gpurun --show-topology
  gpurun -g 4,5 --show-topology

  CUDA_VISIBLE_DEVICES=2,3 gpurun python train.py

With torchrun, use local device ids in -C (0,1,...), not physical GPU numbers.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog=_PROG,
        description=(
            "Set CUDA_VISIBLE_DEVICES and run a command with GPU-local CPU/NUMA "
            "affinity from nvidia-smi topology."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_EPILOG,
    )
    p.add_argument(
        "-g",
        "--gpus",
        dest="gpus",
        metavar="IDS",
        help="Physical GPU indices (sets CUDA_VISIBLE_DEVICES). "
        "Default: CUDA_VISIBLE_DEVICES if set.",
    )
    p.add_argument(
        "--mode",
        choices=("auto", "numactl", "taskset", "none"),
        default="auto",
        help="CPU pinning: auto=numactl then taskset (default), none=skip pinning.",
    )
    p.add_argument(
        "--no-pin",
        action="store_true",
        help="Alias for --mode none (still sets CUDA_VISIBLE_DEVICES when -g is used).",
    )
    p.add_argument(
        "--fallback",
        action="store_true",
        help="If pinning fails in auto mode, run the command unchanged with a warning.",
    )
    p.add_argument(
        "--show-topology",
        action="store_true",
        help="Print parsed GPU → CPU/NUMA mapping (and raw topo with --verbose).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print extra detail: raw nvidia-smi topo with --show-topology; "
        "before launch, print export CUDA_VISIBLE_DEVICES + wrapped command.",
    )
    p.add_argument(
        "--show-command",
        action="store_true",
        help="Before launch, print export CUDA_VISIBLE_DEVICES + wrapped command "
        "(to stderr; command still runs). Same output as --dry-run.",
    )
    p.add_argument(
        "--refresh-topology",
        action="store_true",
        help="Re-query nvidia-smi and update ~/.cache/gpurun/topology.json.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print export + wrapped command without executing.",
    )
    p.add_argument(
        "cmd",
        nargs=argparse.REMAINDER,
        help="Command to run (prefix with -- if it starts with -).",
    )
    args = p.parse_args(argv)
    if args.cmd[:1] == ["--"]:
        args.cmd = args.cmd[1:]
    if not args.show_topology and not args.cmd:
        p.error("command to run is required (or use --show-topology)")
    return args


def resolve_gpu_ids(gpus: str | None) -> list[int]:
    if gpus is not None:
        return parse_gpu_ids(gpus)
    from_env = gpu_ids_from_env()
    if from_env is None:
        raise ValueError("pass -g 4,5 or set CUDA_VISIBLE_DEVICES")
    return from_env


def effective_mode(args: argparse.Namespace) -> PinMode:
    if args.no_pin:
        return "none"
    return args.mode  # type: ignore[return-value]


def env_with_cuda(gpu_ids: list[int]) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices_str(gpu_ids)
    return env


def format_launch_preview(launch: list[str], gpu_ids: list[int]) -> str:
    """Shell-style preview: export line plus wrapped argv."""
    cvd = cuda_visible_devices_str(gpu_ids)
    return f"export CUDA_VISIBLE_DEVICES={cvd}\n{shlex.join(launch)}"


def print_launch_preview(
    launch: list[str],
    gpu_ids: list[int],
    *,
    stream: io.TextIOBase | None = None,
) -> None:
    out = sys.stderr if stream is None else stream
    print(format_launch_preview(launch, gpu_ids), file=out)


def wants_launch_preview(args: argparse.Namespace) -> bool:
    return args.verbose or args.show_command


def show_topology(
    gpu_ids: list[int] | None,
    *,
    refresh: bool,
    verbose: bool,
) -> int:
    snap = get_topology_snapshot(refresh=refresh)
    if verbose:
        sys.stdout.write(snap.raw_topo)
        if not snap.raw_topo.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.write("\n")
    print(format_topology_table(snap))
    print(f"# cache: {topology_cache_path()}", file=sys.stderr)
    if gpu_ids is not None:
        for g in gpu_ids:
            aff = snap.gpus.get(g)
            if aff is None:
                print(f"GPU{g}: (not in topology)", file=sys.stderr)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    mode = effective_mode(args)

    if args.show_topology:
        try:
            gpu_ids = resolve_gpu_ids(args.gpus) if args.gpus else None
        except ValueError as exc:
            print(f"{_PROG}: {exc}", file=sys.stderr)
            return 2
        try:
            return show_topology(
                gpu_ids, refresh=args.refresh_topology, verbose=args.verbose
            )
        except RuntimeError as exc:
            print(f"{_PROG}: {exc}", file=sys.stderr)
            return 1

    try:
        gpu_ids = resolve_gpu_ids(args.gpus)
    except ValueError as exc:
        print(f"{_PROG}: {exc}", file=sys.stderr)
        return 2

    try:
        launch, warning = build_launch_cmd(
            args.cmd,
            gpu_ids=gpu_ids,
            mode=mode,
            refresh_topology=args.refresh_topology,
        )
    except RuntimeError as exc:
        if args.fallback and mode == "auto":
            print(f"{_PROG}: warning: {exc}", file=sys.stderr)
            print(f"{_PROG}: running without CPU pinning (--fallback)", file=sys.stderr)
            launch = list(args.cmd)
            warning = None
        else:
            print(f"{_PROG}: {exc}", file=sys.stderr)
            if mode != "none":
                print(
                    f"{_PROG}: retry with --no-pin, --fallback, or --mode none",
                    file=sys.stderr,
                )
                print(f"{_PROG}: debug: {_PROG} --show-topology", file=sys.stderr)
            return 1

    if warning:
        if args.fallback:
            print(f"{_PROG}: {warning}", file=sys.stderr)
        else:
            print(f"{_PROG}: {warning}", file=sys.stderr)
            print(f"{_PROG}: use --fallback to run anyway", file=sys.stderr)
            return 1

    env = env_with_cuda(gpu_ids)

    if args.dry_run:
        print(format_launch_preview(launch, gpu_ids))
        return 0

    if wants_launch_preview(args):
        print_launch_preview(launch, gpu_ids)

    os.execvpe(launch[0], launch, env)
    return 0  # unreachable


if __name__ == "__main__":
    raise SystemExit(main())
