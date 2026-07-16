# Diffusion Policy: Push-T Low-Dim Paper-to-Code Map

This note connects the paper's main ideas to the checked-in Push-T
low-dimensional implementation. It is based on static code inspection and will
be updated after the official checkpoint is run.

## 1. Core Idea

Diffusion Policy treats robot control as conditional action-sequence
generation. Given recent observations, it starts from a random action
trajectory and repeatedly denoises that trajectory into a coherent plan. The
environment executes only part of the plan, collects a new observation, and
plans again.

## 2. Baseline Parameters

| Paper/code concept | Push-T low-dim value | Configuration source |
|---|---:|---|
| Observation dimension per step | 20 | `config/task/pusht_lowdim.yaml` |
| Observation steps | 2 | `train_diffusion_unet_lowdim_workspace.yaml` |
| Action dimension | 2 | `config/task/pusht_lowdim.yaml` |
| Prediction horizon | 16 | `train_diffusion_unet_lowdim_workspace.yaml` |
| Actions executed per replanning round | 8 | `train_diffusion_unet_lowdim_workspace.yaml` |
| Training diffusion timesteps | 100 | `train_diffusion_unet_lowdim_workspace.yaml` |
| Default inference denoising steps | 100 | `train_diffusion_unet_lowdim_workspace.yaml` |

The 20-dimensional observation consists of 18 keypoint values and 2 agent
state values. The 2-dimensional action represents the planar Push-T control
command used by the environment.

## 3. One Closed-Loop Inference Cycle

```text
two recent observations, each 20-D
                |
                v
normalize observations with checkpoint statistics
                |
                v
create a random [batch, 16, 2] action trajectory
                |
                v
run the scheduler/model denoising loop 100 times
                |
                v
obtain a 16-step, 2-D action prediction
                |
                v
select the configured 8-step execution chunk
                |
                v
step the Push-T environment
                |
                v
collect a new observation and plan again
```

This is receding-horizon control: predict farther ahead than the system
commits to executing, then replan using fresh feedback.

## 4. Paper Concepts and Code Locations

| Concept | Primary code location | Role |
|---|---|---|
| Checkpoint evaluation | `eval.py` | Loads the saved configuration and weights, selects EMA, runs the environment, and writes JSON results |
| Experiment parameters | `diffusion_policy/config/train_diffusion_unet_lowdim_workspace.yaml` | Defines horizon, observation/action steps, scheduler, training, and checkpoint settings |
| Push-T task | `diffusion_policy/config/task/pusht_lowdim.yaml` | Defines dimensions, dataset path, seeds, rollout count, and environment settings |
| Diffusion policy | `diffusion_policy/policy/diffusion_unet_lowdim_policy.py` | Normalizes observations, creates the random trajectory, denoises it, and returns actions |
| Noise predictor | `diffusion_policy/model/diffusion/conditional_unet1d.py` | Predicts the noise or denoising update conditioned on observations and timestep |
| Closed-loop runner | `diffusion_policy/env_runner/pusht_keypoints_runner.py` | Builds parallel environments, calls the policy, executes action chunks, records rewards, and writes videos |
| Training loop | `diffusion_policy/workspace/train_diffusion_unet_lowdim_workspace.py` | Loads batches, computes loss, updates the model and EMA weights, evaluates, and saves checkpoints |

## 5. Reproduction Evaluation Controls

The reproduction branch adds optional controls to `eval.py`. Omitting all of
them preserves the checkpoint's official behavior.

| Option | Purpose |
|---|---|
| `--num-inference-steps` | Compare denoising-step count against latency and task score |
| `--n-envs` | Limit simultaneous rollout environments without reducing the requested seed set |
| `--n-train` | Override the number of training-seed rollouts |
| `--n-test` | Override the number of test-seed rollouts |
| `--seed` | Seed Python, NumPy, PyTorch, and CUDA for a recorded evaluation |
| `--trace-shapes` | Print the first inference call's shapes, step count, and action range |

Effective values are written into `eval_log.json` using `repro/*` keys.

## 6. Training Noise vs. Inference Denoising

### Training

1. Load a demonstrated action sequence.
2. Sample a diffusion timestep and random Gaussian noise.
3. Add the sampled noise to the demonstrated action sequence.
4. Ask the conditional U-Net to predict the noise from the noisy actions,
   timestep, and observations.
5. Compute the loss and update the model.
6. Update the exponential moving average (EMA) model.

Training therefore teaches the network how to remove noise at many noise
levels; it does not repeatedly execute all 100 denoising steps for every
training sample.

### Inference

1. Start from a fully random action trajectory.
2. Set the scheduler to the configured inference timesteps.
3. Call the same noise-prediction model at every denoising step.
4. Convert the final normalized trajectory back to environment action units.
5. Return the configured execution chunk.

Inference latency grows with the number of denoising iterations because the
conditional U-Net is called once per iteration.

## 7. Main Tensor Shapes

For a batch size `B`, the important expected shapes are:

| Tensor | Expected shape | Meaning |
|---|---|---|
| Recent observations | `[B, 2, 20]` | Two observed timesteps |
| Random/predicted trajectory | `[B, 16, 2]` | Sixteen planar actions |
| Returned execution chunk | `[B, 8, 2]` | Actions sent to the environment before replanning |

The shape trace added later must confirm these shapes from a real checkpoint
run rather than relying only on configuration inspection.

## 8. Recommended Code Reading Order

1. `eval.py`: understand the outer evaluation lifecycle.
2. `pusht_lowdim.yaml`: understand the task, dimensions, seeds, and rollouts.
3. `train_diffusion_unet_lowdim_workspace.yaml`: understand 2/16/8/100.
4. `pusht_keypoints_runner.py`: follow observation, policy call, action step,
   reward, and video recording.
5. `diffusion_unet_lowdim_policy.py`: follow normalization, random trajectory,
   denoising loop, and action extraction.
6. `conditional_unet1d.py`: inspect the neural network only after the data flow
   is concrete.
7. `train_diffusion_unet_lowdim_workspace.py`: compare training-time noise
   prediction with inference-time iterative denoising.

## 9. Questions to Verify by Running the System

- Does the official checkpoint restore the expected 2/16/8/100 settings?
- What exact score keys are written to `eval_log.json`?
- How many rollout videos are generated by the resource-limited run?
- Are the observed action tensor shapes consistent with this map?
- What are the single-sample mean, P50, and P95 inference latencies?
- How do 100, 50, 20, 10, and 5 denoising steps change score and latency?
- Which compatibility issues, if any, appear on Ubuntu 24.04?

## 10. Current Boundary

This map does not yet claim that the implementation is reproducible on the
server. Runtime behavior, checkpoint metadata, scores, latency, and videos
must be verified before moving to deeper model-code analysis.
