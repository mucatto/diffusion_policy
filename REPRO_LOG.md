# Diffusion Policy Reproduction Log

Last updated: 2026-07-16

## Current Status

- Stage: local reproduction preparation
- Result: resource-safe evaluation overrides and shape tracing are ready
- Blocking issue: none for local preparation
- Next action: review, commit, and synchronize the local reproduction changes

## Scope and Success Criteria

The first reproduction phase is limited to the official Push-T low-dimensional
pretrained checkpoint. It does not include image-based Push-T, full paper
training, Jetson, TensorRT, a real robot, RealSense, or other benchmarks.

The phase succeeds when:

1. The isolated server environment imports PyTorch and sees one approved GPU.
2. The official checkpoint loads and `eval.py` finishes without a traceback.
3. The evaluation produces a finite score in `eval_log.json`.
4. Rollout videos are generated and visually inspected.
5. Commands, versions, resources, errors, and result paths are recorded here.

## Local Repository

- Path: `E:\MyProject\zty_group\sk\diffusion_policy`
- Baseline commit: `5ba07ac6661db573af695b419a7947ecb704690f`
- Working branch: `repro/pusht-minimal`
- Origin: `git@github.com:mucatto/diffusion_policy.git`
- Upstream: `https://github.com/real-stanford/diffusion_policy.git`

## Remote Server Baseline

Read-only checks were completed before any remote project modification.

- SSH alias: `zty-server`
- SSH user: `zty_group`
- Personal project root: `/home/zty_group/sk`
- Planned project path: `/home/zty_group/sk/diffusion_policy`
- Project status: not cloned yet
- OS: Ubuntu 24.04.3 LTS
- System Python: 3.12.3 at `/usr/local/bin/python`
- Conda/Mamba in `PATH`: not found
- `/home` free space at inspection: approximately 5.5 TiB
- GPU snapshot: GPUs 0 and 7 had no compute process at inspection time
- GPU warning: availability is transient and must be checked again immediately
  before every evaluation or training launch

### Network Baseline

- No HTTP, HTTPS, or SOCKS proxy variables were active in the SSH session.
- Git had no global proxy or URL rewrite.
- Tested traffic used the physical `ens3f0` route, not `tun0` or `wg0`.
- Conda channel alias: USTC mirror
- Pip index: Tsinghua mirror
- GitHub and the official checkpoint host were directly reachable.

## Resource and Safety Limits

- Do not use `sudo`.
- Do not terminate, pause, or interfere with any existing process.
- Do not modify `/home/zty_group/.codex` or
  `/home/zty_group/.vscode-server`.
- Do not access unrelated projects under `/home/zty_group/sk`.
- Use at most one GPU, selected only after a fresh `nvidia-smi` check.
- Planned CPU limit: 6 logical CPUs, below the default ceiling of 8.
- Planned memory hard limit: 12 GiB; stop before launch if it cannot be
  enforced safely for the new process.
- Keep the environment, caches, logs, checkpoints, and videos inside the
  current remote project.
- Do not delete remote files without explicit approval.
- Before every remote modification or experiment, show the exact command,
  affected paths, expected resource use, duration, and output paths.

## Completed Preparation Batches

### Batch 1: Git Preparation

- Added the official repository as local `upstream`.
- Created and switched to `repro/pusht-minimal`.
- No project file was modified by this batch.

### Batch 2: Ignore Rules

Added ignore coverage for:

- `.tools/`
- `runs/`
- `*.ckpt`
- `*.mp4`
- `*.zip`

Existing `data`, `outputs`, and `wandb` rules remain active.

### Batch 3: Learning Documents

- Created this reproduction log.
- Created `docs/paper_code_map.md`.
- Recorded the paper-to-code mapping for Push-T low-dimensional inference.

### Batch 4: Evaluation Controls

Added backward-compatible optional arguments to `eval.py`:

- `--num-inference-steps`
- `--n-envs`
- `--n-train`
- `--n-test`
- `--seed`
- `--trace-shapes`

When these options are omitted, checkpoint configuration remains unchanged.
Effective evaluation settings are written into `eval_log.json` under
`repro/*` keys.

Validation and tests:

- Python compilation: passed
- Unit tests: 8 passed
- `git diff --check`: passed

### Batch 5: Static Compatibility Review

- Confirmed `PushTKeypointsRunner` officially accepts `n_envs`.
- Confirmed `n_envs: null` defaults to `n_train + n_test`.
- Confirmed a smaller `n_envs` value processes all requested rollouts in
  multiple chunks.
- Confirmed the low-dimensional baseline uses horizon 16, two observation
  steps, eight executed action steps, and 100 inference denoising steps.

## Environment Setup

- Status: pending
- Commands: pending approval
- Environment path: pending
- Python: pending
- PyTorch: pending
- CUDA available: pending
- Result: pending
- Errors: none yet

## Pretrained Push-T Low-Dim Evaluation

- Status: pending
- Checkpoint: official `epoch=0550-test_mean_score=0.969.ckpt`
- Device: pending fresh GPU check and approval
- Inference steps: 100 for the official baseline
- Parallel environments: at most 8; planned default is 6 or fewer
- Random seed: fixed and recorded before execution
- Command: pending approval
- Result: pending
- Errors: none yet

## Result Paths

- Setup log: pending
- Evaluation run log: pending
- `eval_log.json`: pending
- Rollout videos: pending
- Extracted video-review frames: pending
- Local inspection copy: pending

## Errors and Resolutions

### Read-only SSH command quoting

- Symptom: two compound inspection commands failed before reaching the remote
  operation because PowerShell and Bash parsed nested quotes differently.
- Resolution: split the inspection into small read-only commands.
- Impact: no remote files, settings, processes, or services were changed.

## Next Action

Review and commit the local reproduction changes. After synchronization, carry
out a fresh read-only remote resource check before proposing environment setup
or checkpoint download commands.
