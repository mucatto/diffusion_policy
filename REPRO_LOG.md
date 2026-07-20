# Diffusion Policy Reproduction Log

Last updated: 2026-07-20

## Current Status

- Stage: official low-dimensional Push-T pretrained evaluation
- Result: isolated Docker environment and checkpoint are ready; the first
  evaluation reached the Python import stage and exposed a dependency mismatch.
- Blocking issue: resolved in a repaired, tagged evaluation-candidate image;
  the official smoke evaluation must be rerun.
- Next action: fresh GPU inspection, then rerun the eight-seed smoke evaluation
  with the repaired image.

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
- Project path: `/home/zty_group/sk/diffusion_policy`
- Project branch and commit after synchronization: `repro/pusht-minimal` at
  `a900d72`
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

- Status: ready for low-dimensional Push-T evaluation.
- Image recipe: `docker/Dockerfile.pusht` and
  `docker/install_pusht_env.sh`.
- Initial image: `zty/diffusion-policy-pusht:torch1.12-cu116`.
- Repaired evaluation-candidate image:
  `zty/diffusion-policy-pusht:torch1.12-cu116-hf0121`.
- Python: 3.10.12.
- PyTorch: 1.12.1+cu116.
- Diffusers: 0.11.1.
- Hugging Face Hub: 0.12.1, pinned in `docker/requirements-pusht.txt`.
- CUDA visibility: verified separately on approved GPU 3 in a restricted
  container; normal evaluation still requires a fresh preflight check.

## Pretrained Push-T Low-Dim Evaluation

- Status: first smoke run stopped before rollout; rerun pending.
- Checkpoint: official `epoch=0550-test_mean_score=0.969.ckpt`
- Device in first attempt: GPU 3 after a fresh check; this selection is stale
  and must not be reused without another check.
- Inference steps: 100 for the official baseline
- Parallel environments: at most 8; planned default is 6 or fewer
- Random seed: fixed and recorded before execution
- First smoke settings: 100 inference steps, four parallel environments, zero
  train rollouts, eight test rollouts, seed `20260720`, one-hour timeout.
- Result: no `eval_log.json` or rollout video because importing Diffusers
  failed before the runner started.

## Result Paths

- Bootstrap logs:
  `/groups/2/sk/diffusion_policy/runs/zty-dp-pusht-bootstrap-20260719-191020.log`
- Checkpoint download log:
  `/groups/2/sk/diffusion_policy/runs/download-pusht-lowdim-retry.log`
- Official checkpoint:
  `/groups/2/sk/diffusion_policy/checkpoints/pusht_lowdim.ckpt`
- First evaluation driver log:
  `/groups/2/sk/diffusion_policy/runs/pusht-lowdim-pretrained-smoke-20260720.driver.log`
- `eval_log.json`: pending successful rerun.
- Rollout videos: pending successful rerun.
- Extracted video-review frames: pending
- Local inspection copy: pending

## Errors and Resolutions

### Diffusers and Hugging Face Hub import mismatch

- Symptom: the first official Push-T smoke evaluation stopped before rollout
  with `ImportError: cannot import name 'HfFolder' from 'huggingface_hub'`.
- Cause: `diffusers==0.11.1` has no upper bound for `huggingface-hub`; pip
  selected 1.16.1, which removed `HfFolder` required by this old Diffusers
  release.
- Resolution: verified `huggingface-hub==0.12.1` in a disposable restricted
  container, pinned it in `docker/requirements-pusht.txt`, and committed a
  separately tagged repaired image. The repair image carries labels for
  project, task, purpose, base image, status, and the exact fix.
- Impact: no GPU rollout ran, no metrics or videos were generated, and the
  public checkpoint was not modified.

### Read-only SSH command quoting

- Symptom: two compound inspection commands failed before reaching the remote
  operation because PowerShell and Bash parsed nested quotes differently.
- Resolution: split the inspection into small read-only commands.
- Impact: no remote files, settings, processes, or services were changed.

## Next Action

Run a fresh read-only resource check. If an approved GPU remains free, rerun
the eight-test-rollout official smoke evaluation using
`zty/diffusion-policy-pusht:torch1.12-cu116-hf0121`.
