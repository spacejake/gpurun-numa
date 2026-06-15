# 🚀 gpurun-numa Feature Roadmap

### 🔍 1. Smart Auto-Discovery Mode
- [ ] Implement `-g auto` and `-g auto:N` flags to automatically find free GPUs via `nvidia-smi`.
- [ ] Parse local cached topology to ensure multi-GPU requests (for DDP) are grouped on the *same* physical NUMA node.
- [ ] Add a tie-breaker heuristic to choose the NUMA node with the lowest overall CPU/Memory utilization.

### 🛑 2. Conflict Guardrails
- [ ] Read VRAM usage of explicitly targeted GPUs (e.g., `-g 0,1`) right before execution.
- [ ] Block the launch and print a warning if a targeted GPU is already heavily occupied by another process.
- [ ] Add a `--force` flag to override the block in case users intentionally want to share a GPU.

### 📊 3. Automated Notion Syncing
- [ ] Add support for reading a `NOTION_API_KEY` and `DATABASE_ID` from a local configuration file (e.g., `~/.config/gpurun-numa/config`).
- [ ] Send a `PATCH` request to the lab's Notion dashboard on script startup to log `Server`, `GPU ID`, `User`, and `Status: Running`.
- [ ] Implement exit hooks (`trap` in Bash or `try/finally` in Python) to automatically clear the user's name and reset status to `Free` upon job completion or crash.

### ⚡ 4. Ergonomic Resource Profiles
- [ ] Create simple profile presets (e.g., `--size small|medium|large`) that map to predefined fractions of a NUMA node's CPU/RAM.
- [ ] Automatically calculate the best `--physcpubind` core ranges based on these profiles so users don't have to manually count threads.
