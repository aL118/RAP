# RL fine-tuning for RAP

PPO fine-tuning of the RAP planner against the NAVSIM PDM score, built on
stable-baselines3. Reads exactly the caches the supervised pipeline reads, but
treats the PDM scorer as an environment: the policy proposes a trajectory, the
scorer returns a reward, collisions are punished.

## Formulation

**One episode is one scene and exactly one step.** That is not a shortcut, it is
what the data supports. The metric cache is a frozen 4-second snapshot of a
scenario — observation, centerline, drivable area, and the *recorded* futures of
every other agent. Nothing in it is reactive, so there is no way to advance the
world and re-observe. The problem is therefore a contextual bandit: observe a
scene, emit a whole trajectory, receive its score. SB3 handles this with no
special casing (see `gamma` below).

| | |
|---|---|
| **Observation** | frozen scene latent + ego status + RAP's own trajectory and predicted scores |
| **Action** | `Box(-1, 1, (num_poses * 3,))` — a bounded residual on RAP's trajectory |
| **Reward** | `r(policy) - r(base)`, where `r` weights PDM sub-scores and penalises collisions |
| **Episode** | 1 step, always `terminated=True` |

### Why a residual, not a trajectory from scratch

The pretrained RAP planner already scores ~0.85–0.97 PDMS. Learning to drive from
a random initialisation would spend the entire budget rediscovering that. The
policy instead emits a bounded delta on RAP's own output, ramped linearly over the
horizon so pose 0 barely moves and the trajectory stays anchored at the ego. A
freshly initialised policy reproduces RAP exactly, and every reward it sees is
"did I improve on RAP, on this scene".

### Why the observation is precomputed

A cached RAP sample carries 4 camera images of 3×448×768 — 16.5 MB of float32. An
on-policy rollout buffer of even a few thousand is hundreds of GB. The DINOv3
backbone is frozen in RAP training anyway, so running it inside the RL loop would
recompute an identical tensor every epoch. `precompute.py` runs the pretrained
model over the cache **once** and keeps ~21 KB per token instead of 16.5 MB. The
whole 14,675-token navtrain subset then fits in ~300 MB of RAM and the env runs no
neural network at all.

The trade-off is explicit: **perception is frozen.** The policy adjusts planning,
not what the model sees.

## Run order

```bash
# 0. preflight, ~10 min: observation parity + a 32-token precompute.
#    Catches the silent failures below before they cost a GPU-day.
sbatch scripts/rl/run_rl_precompute_check.sh

# 1. once, on a GPU (~1-2 h for 14675 tokens: ViT inference + PDM scoring the baselines)
sbatch scripts/rl/run_rl_precompute.sh

# 2. check the reward actually has gradient before spending a GPU-day on it
python rl/eval.py --probe --probe-scenes 40

# 3. train, then score the held-out split against the pretrained planner
sbatch scripts/rl/run_rl_train.sh

# 4. the number that is comparable to anything: navhard EPDMS, through the devkit.
#    RUN_BASELINE=1 scores plain RAP through the identical code path first.
RUN_BASELINE=1 sbatch scripts/rl/run_rl_navhard_eval.sh
```

Step 2 is worth the two minutes. It reports the mean PDMS cost of fixed
perturbations to RAP's trajectory and flags any that are **flat**, i.e. directions
in which the policy would receive no gradient at all:

```
perturbation       mean PDMS     vs base  collision rate
--------------------------------------------------------
base                  0.9711     +0.0000          0.0000
lon x1.5              0.6153     -0.3558          0.1250
left 1m               0.9609     -0.0102          0.0000
right 1m              0.8008     -0.1703          0.0000
right 4m              0.4532     -0.5179          0.3000
```

`residual_scale` in `config.py` is sized from exactly this table: 1 m is large
enough that the reward moves, small enough that saturating the action cannot by
itself cause a collision.

## Three things that will silently ruin a run

**1. Never parallelise the scorer with threads.**
`compute_navsim_score` builds a module-level `PDMSimulator`/`PDMScorer` pair, and
`PDMScorer` keeps per-call state on `self` (`_multi_metrics`, `_ego_areas`, …).
Concurrent calls in one process interleave and silently corrupt each other's
results — no exception, just wrong rewards. Use `SubprocVecEnv` (processes), which
is what `train.py` does. This is also why `n_envs` is a **CPU** count: the policy
is a small MLP, and the bottleneck is the scorer at ~0.2 s per trajectory.

**2. The reward must be centred on the per-scene baseline.**
Most navtrain scenes are collision-free, so raw rewards inside a PPO minibatch are
nearly identical (~0.97 ± 0.01). SB3 normalises advantages by their minibatch
standard deviation, and dividing near-identical returns by a near-zero std
amplifies pure noise into an enormous policy gradient. Measured on the first
updates of a run with `relative_reward=False`:

| | raw reward | centred on base |
|---|---|---|
| `approx_kl` | 73109 | 0.05 |
| `clip_fraction` | 0.92 | 0.21 |

`relative_reward=True` (the default) subtracts the base trajectory's own score,
removing the shared component that carries no information about the action.
`precompute.py` scores every base trajectory once, so a rollout step still makes
exactly one PDM call. `target_kl=0.05` is a backstop, not the fix.

**3. The agent's observation must match the one the policy trained on.**
`precompute.py` builds observations for training; `agent.py` rebuilds them live at
benchmark time. They must agree elementwise — same fields, same order, same flattening,
same fp16 rounding — and then be normalised with the *training* run's `vecnormalize.pkl`.
Every one of those is silent if wrong: the policy still emits actions, the trajectories
still score, and the benchmark simply reports that RL did not help, which is
indistinguishable from a real negative result. `rl/parity_check.py` asserts it directly,
and `rl/residual.py` is shared by the env and the agent so the action→trajectory map
physically cannot drift.

## Two evaluations, and only one is comparable

`rl/eval.py` and `scripts/rl/run_rl_navhard_eval.sh` answer different questions, and
mixing them up is the easiest way to report a number that means nothing.

| | `rl/eval.py` | `run_rl_navhard_eval.sh` |
|---|---|---|
| split | held-out 5% of navtrain | `navhard_two_stage` |
| input | the precomputed cache | live sensors, through the devkit |
| metric | v1 PDMS — the training reward | EPDMS |
| runs | seconds | hours |
| use | did PPO learn anything? | is the result worth reporting? |

`eval.py` reuses the frozen latents, the frozen base trajectories and the train-split
metric cache the policy trained on. That makes it fast and exactly aligned with the
reward, which is what you want while iterating — and also means it cannot be compared
to any published number, including RAP's own.

The benchmark script runs the devkit's `run_pdm_score.py` with `agent=rl_agent`, so the
policy is measured by the identical code that produced the baseline. Note that **the RL
reward is v1 PDMS while navhard reports EPDMS**, which adds sub-scores the reward never
saw (two-frame extended comfort, lane keeping) plus a second stage of synthetic scenes.
Improving the reward therefore does not automatically improve the headline metric. Run
`RUN_BASELINE=1` so both columns come from the same code path — `disable_residual=True`
is plain RAP with the residual forced to zero, which cancels any difference in feature
building or proposal selection and leaves the residual as the only variable.

## Files

| file | role |
|---|---|
| `config.py` | every path and hyper-parameter, one dataclass |
| `reward.py` | PDM sub-scores → scalar reward |
| `residual.py` | action → trajectory, shared by the env and the agent |
| `precompute.py` | RAP cache → compact RL observations (GPU, run once) |
| `env.py` | `RAPPlanningEnv`, a `gymnasium.Env` |
| `train.py` | PPO entry point |
| `eval.py` | policy vs. pretrained RAP on held-out navtrain, and `--probe` |
| `agent.py` | `RLResidualAgent` — the policy as a NAVSIM agent, for the benchmark |
| `parity_check.py` | asserts `agent.py` sees exactly what `precompute.py` wrote |

## Notes

- **`gamma=0.0`** is deliberate. Episodes terminate after one step, so there is
  nothing to bootstrap and the advantage is `r - V(s)` for any gamma; 0.0 says so.
- **`log_std_init=-2.0`** (std ≈ 0.135). SB3's default std of 1.0 would make the
  first residuals full-scale — metres — throwing away RAP's pretrained trajectory
  before PPO sees a single gradient.
- **`vecnormalize.pkl`** is written next to `final_model.zip` and is part of the
  trained model. `scene_latent` is 5120 raw DINOv3 activations; feeding a policy
  unnormalised observations it was not trained on looks exactly like a training
  failure. `eval.py` refuses to run without it.
- **stable-baselines3 2.6.0**, installed with `--no-deps`. `ray[rllib]` in this env
  pins `gymnasium==1.1.1`, and letting pip resolve SB3's own bound would downgrade
  it. 2.6.0 declares `torch>=2.3` but runs correctly on this env's torch 2.1.
- **Every worker pool in this package must use `spawn`.** `precompute.py` runs the RAP
  model on the GPU and then forks a pool to PDM-score the base trajectories. Forking an
  initialised CUDA context deadlocks: observed in job 7435020 as all 8 workers sitting in
  S state at 0.0% CPU, holding the parent's 2.7 GB image, having scored nothing — no
  exception, no progress, just a job that has to be cancelled. Same reason `train.py`
  spawns `SubprocVecEnv`.
- **`obs_meta.json` pins the base model.** The residual is a delta on whatever
  trajectory RAP emits, so the benchmark must use the checkpoint the observations were
  built from — and the supervised pipeline's eval script defaults to a *different* one.
  `precompute.py` records the checkpoint in `meta.json`, `train.py` copies it next to
  the model as `obs_meta.json`, and `agent.py` refuses to run if the sizes disagree.
- **6 of the 30 action dimensions are inert.** The PDM scorer truncates to 8 poses
  (`get_sub_score`: `Trajectory(model_trajectory[:8])`) and `agent.py` truncates the
  same way, so the residual on poses 8–9 cannot change the reward at either end. They
  are consistent, not wrong — but the policy still samples noise into them and it
  counts towards `approx_kl`. See the note on `num_poses` in `config.py` for why
  shrinking the action space was left as a deliberate decision rather than a quiet fix.
- The metric cache is **agent-independent** — it stores the scenario's simulation
  and scoring state, and any trajectory is scored against it afterwards. Reused
  from DrivoR rather than rebuilt.
