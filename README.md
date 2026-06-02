# gpurun-numa

Run any command with **GPU-local CPU and NUMA affinity** on Linux NVIDIA hosts.

`gpurun` parses `nvidia-smi topo -m`, picks the CPUs and memory nodes closest to your selected GPUs, sets `CUDA_VISIBLE_DEVICES`, and wraps your command with `numactl` or `taskset`.

> **Install now:** `pip install "gpurun-numa @ git+https://github.com/spacejake/gpurun-numa.git"`  
> **PyPI:** distribution name [`gpurun-numa`](https://pypi.org/project/gpurun-numa/) — `gpurun` on PyPI is a different tool. CLI command: **`gpurun`**.

## Why this matters

Training jobs that pin dataloader workers and the main process to the wrong socket pay a large penalty: PCIe hops, QPI/UPI traffic, and noisy neighbors on remote NUMA nodes. Binding to the CPUs NVIDIA reports for each GPU keeps H2D copies and CPU-side preprocessing on the local socket.

## Requirements

- Linux (tested on servers with `nvidia-smi topo -m`)
- NVIDIA driver + `nvidia-smi`
- Optional: `numactl` (preferred) or `taskset` from util-linux
- No root required

## Install

### From GitHub (recommended before PyPI release)

```bash
pip install "gpurun-numa @ git+https://github.com/spacejake/gpurun-numa.git"
```

Pin a branch or tag:

```bash
pip install "gpurun-numa @ git+https://github.com/spacejake/gpurun-numa.git@main"
```

Editable (live checkout for development):

```bash
git clone https://github.com/spacejake/gpurun-numa.git
cd gpurun-numa
pip install -e .
```

With **uv**:

```bash
uv pip install "gpurun-numa @ git+https://github.com/spacejake/gpurun-numa.git"
```

Use in another project’s `pyproject.toml`:

```toml
dependencies = [
    "gpurun-numa @ git+https://github.com/spacejake/gpurun-numa.git",
]
```

### From PyPI (after release)

```bash
pip install gpurun-numa
gpurun --help
```

### From source (local checkout)

```bash
cd /path/to/gpurun
pip install .
# or
uv pip install .
# editable
pip install -e .
```

## Usage

### Single GPU

```bash
gpurun -g 4 python train.py --config config.yaml
```

Sets `CUDA_VISIBLE_DEVICES=4` and runs:

```bash
numactl --physcpubind=<gpu4-cpus> --membind=<numa> python train.py ...
```

### Multi-GPU + torchrun (DDP)

```bash
gpurun -g 4,5 torchrun --standalone --nproc_per_node=2 \
  train.py --config config.yaml
```

Use **local** rank ids in torchrun (`-C 0,1`), not physical GPU numbers — `gpurun` remaps devices via `CUDA_VISIBLE_DEVICES`.

### From existing `CUDA_VISIBLE_DEVICES`

```bash
export CUDA_VISIBLE_DEVICES=2,3
gpurun python train.py
```

### Dry run / preview launch command

```bash
gpurun -g 4,5 --dry-run torchrun --standalone --nproc_per_node=2 train.py
```

Prints (stdout) and does not execute:

```text
export CUDA_VISIBLE_DEVICES=4,5
numactl --physcpubind=12-23,36-47 --membind=1 torchrun ...
```

Print the same preview and **still run** (stderr, then exec):

```bash
gpurun -g 4,5 --show-command python train.py
gpurun -g 4,5 --verbose python train.py   # same launch preview
```

### Show topology

```bash
gpurun --show-topology
gpurun -g 4,5 --show-topology
gpurun --show-topology --verbose   # include raw nvidia-smi topo -m
```

Parsed mapping is cached at `~/.cache/gpurun/topology.json` (refreshed when hostname or GPU count changes, or with `--refresh-topology`).

## Options

| Flag | Description |
|------|-------------|
| `-g`, `--gpus` | Physical GPU indices (sets `CUDA_VISIBLE_DEVICES`) |
| `--mode auto` | `numactl` if available, else `taskset` (default) |
| `--mode numactl` | Require `numactl --physcpubind` + `--membind` |
| `--mode taskset` | Require `taskset -c` |
| `--mode none` | No CPU pinning |
| `--no-pin` | Same as `--mode none` |
| `--fallback` | In `auto` mode, run without pinning if tools/topology fail |
| `--dry-run` | Print `export` + wrapped command (stdout), do not execute |
| `--show-command` | Print `export` + wrapped command (stderr), then execute |
| `--show-topology` | Print GPU → CPU/NUMA table |
| `--refresh-topology` | Re-query `nvidia-smi` and update cache |
| `--verbose` | Raw topo with `--show-topology`; launch preview with a command (like `--show-command`) |

## Limitations

- **Linux-focused** — relies on `nvidia-smi` and optional `numactl`/`taskset`
- **NVIDIA GPUs only**
- Some consumer boards report `N/A` for CPU Affinity in `topo -m`; use `--no-pin` or `--fallback`
- Cache is per-machine (`hostname` + GPU count); use `--refresh-topology` after hardware changes

## Development

```bash
pip install -e ".[dev]"
pytest
```

Override cache directory in tests:

```bash
GPURUN_CACHE_DIR=/tmp/gpurun-test pytest
```

## Publishing

See [docs/PUBLISHING.md](docs/PUBLISHING.md) for building wheels and uploading to TestPyPI / PyPI (`python -m build`, `twine upload`). CI publishes on GitHub Release via `.github/workflows/publish.yml` (trusted publishing).

## License

Apache-2.0 — see [LICENSE](LICENSE).
