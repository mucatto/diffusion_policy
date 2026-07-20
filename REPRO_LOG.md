# Diffusion Policy Reproduction Log

Last updated: 2026-07-20

## Current Status

- Stage: official low-dimensional Push-T pretrained evaluation completed
- Result: the validated environment loaded the official checkpoint, completed
  eight test rollouts, produced finite metrics, and generated valid videos.
- Blocking issue: none for the first reproduction phase.
- Next action: follow one inference call through the core code, then prepare the
  bounded low-dimensional short-training run from the four-week plan.

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
- Project branch and environment-recipe commit: `repro/pusht-minimal` at
  `2ddc935`
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
- Container memory hard limit: 10 GiB, below the workspace ceiling of about
  12 GiB.
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

- Status: validated by imports, MP4 encode/decode, checkpoint loading, and a
  complete eight-test-rollout evaluation.
- Image recipe: `docker/Dockerfile.pusht` and
  `docker/install_pusht_env.sh`.
- Initial image: `zty/diffusion-policy-pusht:torch1.12-cu116`.
- Validated image: `zty/diffusion-policy-pusht:lowdim-v1`.
- Image ID: `sha256:d2c990d6f966b3663a3bc3b4df69d595449335671646c78253c7e45fffef2930`.
- Image size: 5,767,462,964 bytes; its new writable environment layer was
  approximately 371 MiB and shares all unchanged parent layers.
- Python: 3.10.12.
- PyTorch: 1.12.1+cu116.
- Diffusers: 0.11.1.
- Hugging Face Hub: 0.12.1, pinned in `docker/requirements-pusht.txt`.
- Pandas: 1.5.3.
- PyAV: 10.0.0, built with Cython 0.29.36 against Ubuntu 22.04 FFmpeg 4.4.
- CUDA visibility and inference were verified on an idle RTX 4090 selected by
  a fresh preflight check. Future runs still require a new check.
- The committed image inherits `/bin/bash` from its parent as an entrypoint;
  the evaluation wrapper explicitly overrides it with `/usr/bin/timeout`.

## Pretrained Push-T Low-Dim Evaluation

- Status: passed.
- Checkpoint: official `epoch=0550-test_mean_score=0.969.ckpt`
- Device: GPU 3, an RTX 4090 with no compute process at launch.
- Inference steps: 100 for the official baseline
- Parallel environments: at most 8; planned default is 6 or fewer
- Random seed: `20260720`.
- Successful smoke settings: 100 inference steps, four parallel environments, zero
  train rollouts, eight test rollouts, seed `20260720`, one-hour timeout.
- Result: `test/mean_score = 0.9994508049019712`; all eight per-seed scores
  were finite and ranged from 0.9965998055 to 1.0.
- Runtime shape trace: observations `[4, 2, 20]`, denoised trajectory
  `[4, 16, 2]`, and returned execution chunk `[4, 8, 2]`.
- Four parallel-environment videos were generated. Their decoded contact
  sheets showed the gray T moving into the green target region without black
  frames or corrupt output.

## Result Paths

- Bootstrap logs:
  `/groups/2/sk/diffusion_policy/runs/zty-dp-pusht-bootstrap-20260719-191020.log`
- Checkpoint download log:
  `/groups/2/sk/diffusion_policy/runs/download-pusht-lowdim-retry.log`
- Official checkpoint:
  `/groups/2/sk/diffusion_policy/checkpoints/pusht_lowdim.ckpt`
- Checkpoint SHA-256:
  `f804e16575e261fa0b7e981da3f67741fc8517817734320d550e43a4182bf876`
- First evaluation driver log:
  `/groups/2/sk/diffusion_policy/runs/pusht-lowdim-pretrained-smoke-20260720.driver.log`
- Successful evaluation log:
  `/groups/2/sk/diffusion_policy/runs/pusht-lowdim-pretrained-smoke-lowdim-v1-20260720.log`
- `eval_log.json`:
  `/groups/2/sk/diffusion_policy/runs/pusht-lowdim-pretrained-smoke-lowdim-v1-20260720/eval_log.json`
- Rollout videos:
  `/groups/2/sk/diffusion_policy/runs/pusht-lowdim-pretrained-smoke-lowdim-v1-20260720/media/`
- Local ignored inspection copy and contact sheets:
  `.tools/video-review/pusht-lowdim-v1-20260720/`

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

### Missing low-dimensional evaluation dependencies

- Symptom: subsequent imports reported missing `pandas`, then missing `av`.
- Cause: the initial minimal pip dependency list omitted the JSON logger and
  video-recorder dependencies that are imported by the official evaluation
  path.
- Resolution: pinned pandas 1.5.3 and PyAV 10.0.0. PyAV 10 requires Cython
  0.29.36 and `--no-build-isolation` when compiled in this Python 3.10 image;
  the necessary FFmpeg development libraries are installed only inside the
  container image.
- Verification: `pip check`, full core-module imports, a five-frame MP4
  encode/decode test, checkpoint loading, and the full smoke evaluation passed.

### Read-only SSH command quoting

- Symptom: two compound inspection commands failed before reaching the remote
  operation because PowerShell and Bash parsed nested quotes differently.
- Resolution: split the inspection into small read-only commands.
- Impact: no remote files, settings, processes, or services were changed.

## Next Action

Trace a single inference call through `eval.py`, the runner, policy, scheduler,
and action extraction using the successful runtime evidence recorded here.
