# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-05-29

### Added

- `gpurun` CLI: set `CUDA_VISIBLE_DEVICES` and wrap commands with GPU-local CPU affinity
- Parse `nvidia-smi topo -m` for per-GPU CPU ranges and NUMA nodes
- `--mode auto|numactl|taskset|none` pinning (`numactl --physcpubind` preferred)
- `--fallback`, `--dry-run`, `--show-topology`, `--refresh-topology`
- Topology cache at `~/.cache/gpurun/topology.json`

[0.1.0]: https://github.com/spacejake/gpurun-numa/releases/tag/v0.1.0
