# Iterative scorer training for RAP

Rounds of *collect, score, train* that distil the true PDM score into RAP's own
`Scorer` head, on CARE crash clips with navtrain mixed in.

RAP already trains its scorer against the true PDM score, but only on the
proposals the model happened to emit at that step, scored inline, once, and thrown
away. This replaces that with rounds and a memory.

```
round 1   for every clip window: the human trajectory + a subsample of the
          pretrained model's 64 proposals.  Score all of them with the true
          scorer.  Put them in a buffer.  Train the scorer on the buffer.
round r   sample fresh trajectories from the model round r-1 produced.  Score
          them.  APPEND to the buffer -- nothing is replaced.  Retrain on
          everything.  Repeat, 3-5 rounds.
```

## Why each part is the way it is

**Why rounds at all.** A scorer trained on one model's proposals is only
calibrated on that model's proposals. Train on it, the model's ranking changes,
and the trajectories it now wants to drive are ones the scorer was never asked
about. Sampling fresh each round is the fix; it is DAgger, with the PDM scorer as
the expert.

**Why the buffer never drops anything.** The other half of that bargain. Relabel
from scratch each round and the scorer forgets the trajectories it used to
propose, which is the standard way an iteratively retrained ranker walks off a
cliff. Keeping them is also free: a trajectory's true score does not change when
the model does, so a row scored once is scored forever.

**Why "sampling" means subsampling.** RAP is not stochastic. One forward pass
emits 64 proposals and a predicted score for each, and inference takes the argmax.
So a round takes one pass and splits its budget between the top-ranked proposals
(the on-policy part -- the ones that would actually be driven) and a uniform draw
from the rest (the coverage part). Only the top-k, and the scorer never learns why
it rejects what it rejects. Only uniform, and the rows that matter are a vanishing
fraction of the buffer.

**Why the human trajectory goes in.** It is a real trajectory with a real score,
and on a crash clip usually the most informative row in its window. It is added
once, in round 1 -- it does not depend on the model, so re-adding it every round
would upweight it by a factor of `num_rounds` for nothing.

**Why the human trajectory is not an imitation target on CARE.** A CARE clip is a
crash. Its human trajectory is the one that hit something. It is scored, buffered
and ranked like any other candidate; it is never something the planner is trained
to reproduce. The imitation loss applies to the navtrain half only.

**Why navtrain is mixed in at all.** Without it the scorer sees nothing but crash
footage, and the cheapest way to score well on crash footage is to prefer
trajectories that barely move. The navtrain rows carry ordinary driving, where
progress is rewarded, and they are the only source of an imitation signal. That is
what `regular_ratio` buys, and turning it to 0 is how the model goes timid.

**What the score loss can actually move.** `Bev_refiner` takes the trajectory in
as `ref_2d = pose.detach()`, so no gradient from the score loss reaches a
`traj_decoder`. The score loss trains the Bev_refiner stack, `hist_encoding`,
`init_feature` and the scorer head; the decoders that emit the proposals are
reachable only through the imitation term. Verified on round 1 of the smoke run
(no navtrain, so no imitation loss): every stage's `Bev_refiner` weights moved and
every stage's `traj_decoder` weights came out bit-identical to the pretrained
checkpoint.

Proposals still differ between rounds -- the score loss moves the features the
decoders read -- so the DAgger premise holds. But nothing pulls proposal geometry
towards higher true PDMS on CARE; the scorer learns to rank and inference picks
the best of what was proposed. Read a flat `pool_best` in that light rather than as a
failure. It also means `regular_ratio 0` is not merely "timid": with no imitation
loss the proposal decoders get exactly zero gradient, which is fine for a smoke
test and is not a training configuration.

## Scoring a trajectory the model did not propose

This is the one architectural thing the design needed. `Scorer` does not take a
trajectory -- it takes the refiner stack's query features and pools them. The
trajectory only enters through `Bev_refiner`, which uses the poses as *reference
points* for its deformable attention. So:

```
F_in        = query features entering the last Traj_refiner stage
F_candidate = Bev_refiner(candidate_trajectory, F_in, image_feature)
logits      = pred_score(pool(F_candidate))
```

When `candidate` is the model's own final proposal this reproduces
`RAPModel.forward` exactly, which is what makes it the right hook: the scorer is
trained through the same path it is used through at inference. `rl/model.py` is
that; `RAPScorer.forward_train` returns the proposals and the candidate logits
from one encode, because a training step needs both and the ViT is most of the
cost.

Verified directly: feeding the model its own proposals back through this path
reproduces `propose`'s scores to 0.0, on a CARE window with the pretrained
checkpoint.

The refiner has exactly `proposal_num` (64) query slots, so a scene's candidates
are padded up to 64 by tiling and sliced back afterwards. Scoring is weakly
context-dependent -- temporal self-attention lets query slots see each other, so
scoring 5 candidates padded to 64 differs from scoring the same 5 inside the full
proposal set by 0.0037 predicted PDMS, against a 0.49-0.88 spread across
proposals. Small, and the reason the slot count is held fixed rather than sized to
the batch. Four rounds at 33 rows a
scene overflows that, so a step samples up to `max_candidates` of a scene's rows;
over epochs every row is seen, and the sampling is uniform over rounds -- an old
row is exactly as likely to be trained on as a new one.

## Two scoring backends, and one column that has no ground truth

| | navtrain | CARE |
|---|---|---|
| source | full nuPlan metric cache | boxes and ego poses from the clip export |
| scorer | `compute_navsim_score`, LQR-simulated | direct pose evaluation, b2d style |
| collision, TTC | yes | yes |
| progress | against the map centerline | against the human's own path |
| comfort | nuPlan's six bounds on 0.1 s states | acceleration and yaw rate at nuPlan's thresholds |
| drivable area | yes | **no label at all** |

A CARE clip has no map: `map_location` is the clip's own name and `roadblock_ids`
is empty. The lane polylines in the export are not a substitute -- they are
per-frame detections with no temporal association and a known over-extension bug,
and a confidently wrong label is strictly worse than a missing one because nothing
downstream can tell. So every row carries a `(6,)` mask of which columns are
labelled, and the BCE loss masks `drivable_area_compliance` on CARE rows. The
navtrain half of the buffer is where that column's supervision comes from.

On CARE, a candidate is only charged for a collision **the human did not also
have**. These are crashes; without that subtraction, every candidate anywhere near
the human's path scores zero on collision and the column is a constant. Subtracting
asks the question that matters: did this trajectory hit something the human avoided?

## Run order

```bash
# smoke test first: 4 CARE scenes, no navtrain, its own experiment name so the
# real buffer is never contaminated by a capped round.  Runs on CPU in ~5 min.
python rl/collect.py --round 1 --limit-scenes 4 --samples-per-scene 6 \
                     --batch-size 2 --regular-ratio 0 --score-workers 2 \
                     --experiment-name rap_iterative_smoke
python rl/train.py   --round 1 --epochs-per-round 1 --batch-size 2 \
                     --regular-ratio 0 --experiment-name rap_iterative_smoke
# ...and the two paths a round does not exercise.  --regular-ratio 0 matters
# here too: without it eval builds the navtrain source and asks for a metric
# cache this experiment never used.
python rl/eval.py    --round 1 --experiment-name rap_iterative_smoke \
                     --regular-ratio 0 --limit 3 --proposals-per-scene 4 \
                     --score-workers 1 --device cpu
python rl/export.py  --round 1 --experiment-name rap_iterative_smoke \
                     --out /tmp/round1_full.ckpt

# the invariants behind eval's sampling, which are not visible by inspection
python rl/test_eval_sampling.py

# the pretrained planner's numbers, through this exact code path
python rl/eval.py --round 0 --limit 64

# the real thing: collect -> train -> repeat, resumable at round granularity
python rl/loop.py

# did it help
python rl/eval.py --baseline --round 4

# a checkpoint anything outside this pipeline can load (see below)
python rl/export.py --round 4 --out weights/rap_round4.ckpt
```

`rl/eval.py` reports three numbers per source on scenes no round trained on:

| | |
|---|---|
| `selected` | true PDMS of the proposal the model's own scorer ranked first. The headline. |
| `pool_best` | true PDMS of the best proposal in a **uniform sample** of the proposal set: what the planner has to offer, independent of how it ranks. Rising means the refiners improved. |
| `spearman` | rank correlation between predicted and true score over that same uniform sample. The scorer on its own, independent of proposal quality. |

`pool_best` is **not a ceiling**, and `selected` sitting well above it is the
expected result rather than a bug -- a scorer that could not beat a handful of
random draws would have no reason to exist. Measured on 3 CARE val scenes at
`--proposals-per-scene 4`, round 1 of the smoke run gives `selected` 0.636 against
`pool_best` 0.421 (`spearman` 0.933 -- on 3 sampled proposals per scene, which is
a smoke-test budget and not a number to read into). Only at
`--proposals-per-scene 64` does the uniform sample
become the whole proposal set, and only there is it the true oracle with
`selected <= pool_best`.

Only `selected` is allowed to depend on the scorer being evaluated. Scoring all 64
proposals per val scene is too expensive, so a subset is truly scored -- and if
that subset were the top-k under the checkpoint's own predicted score, a better
scorer would surface better trajectories into it and lift `pool_best` with the
proposals unchanged, while `spearman` would be measured over a differently
restricted range. Round 4 would beat round 0 on both without a single proposal
having improved. So the budget is the model's argmax (keeping `selected` exact)
plus a uniform draw over the query slots, seeded by the scene token: the same
selection rule for every checkpoint. `--proposals-per-scene` sets it.

None of them is comparable to a published NAVSIM number. The split is a held-out
slice of the training sources, and half of it is CARE footage with one sub-score
unlabelled. Always run `--baseline`, so the delta comes from one code path.

## Two patches this package applies to `navsim/` at runtime

Both are in `rl/model.py`, both are needed only because a CARE window has one
camera where navtrain has four, and they are **a pair** -- either alone is worse
than neither.

1. `patch_image_encoder_camera_embeddings`. `ImgEncoder.forward` adds
   `cams_embeds[:, None, None, :]` to a `(num_cam, bs, hw, C)` feature. With one
   camera that does not fail -- it *broadcasts*, replicating the single image four
   times, each copy tagged with a different camera's embedding. `point_sampling`
   then only ever attends to index 0, which in
   `LoadMultiViewImageFromFiles`'s order `(cam_b0, cam_f0, cam_l0, cam_r0)` is the
   **rear** camera. So without this patch a CARE front image is fed through the
   model wearing the back camera's learned embedding, and three quarters of the
   attention compute is spent on copies nothing reads. A four-camera batch takes
   the original method unchanged.

2. `patch_spatial_cross_attention_num_cams`. Once (1) stops replicating, the batch
   genuinely has one camera, and `SpatialCrossAttention`'s hardcoded `num_cams=4`
   reshape asks for four times the elements it has. Nothing about the weights
   depends on the camera count, so the wrapper uses the count actually present.

The real home for both is `navsim/agents/rap_dino/bevformer/`; they live here
because this package does not edit that tree.

## Checkpoints are deltas

A round writes only the tensors it could change and records the pretrained base it
sits on. With `freeze_backbone` the DINOv3 tower is 95% of the model, so that is
~190 MB per round instead of 3.5 GB -- 760 MB for a four-round run rather than
14 GB of a shared filesystem -- and the frozen tower is identical across rounds by
construction rather than by four copies happening to agree.

`RAPScorer.from_checkpoint` merges base and delta transparently, so nothing inside
this package notices. Nothing *outside* it would: every loader on the devkit's path
uses `strict=False`, so a delta handed to `run_pdm_score.py` would be accepted in
silence with a randomly initialised ViT behind it. `rl/export.py` writes a merged,
self-contained checkpoint in the Lightning key layout for exactly that case;
`--save-full` on `rl/train.py` skips the delta entirely.

## Where a round starts from

By default round *r* continues from round *r-1*'s weights, which is cheaper and is
what DAgger normally does. `--restart-from-pretrained` re-fits the pretrained
checkpoint on the whole buffer every round instead: slower, but the rounds stop
compounding. Round 4 is then "the pretrained planner fitted to four rounds of
data" rather than "four fine-tunes stacked on each other", and a bad round cannot
poison the ones after it. Worth reaching for if `selected` starts falling between
rounds while the training loss keeps dropping.

## Watching a run

Every run writes two sinks under its own directory, both on by default:

```
exp/<run>/logs/train.jsonl    one line per batch and per epoch, appended
exp/<run>/logs/eval.jsonl     one line per source per evaluated round
exp/<run>/tb/{train,eval}/    the same numbers as event files
```

```bash
tensorboard --logdir exp/rap_iterative_quicktest/tb    # both sinks, as two runs
```

The JSONL is the durable record and the event files are a view of it. On a cluster
that ordering is the point -- `jq` over the file works from a login node with no
port forwarding, and the file is line-buffered so `tail -f` follows a running job:

```bash
# the two sources' loss curves side by side, which is the thing an epoch mean hides
jq -r 'select(.kind=="batch") | [.step, .source, .loss, .grad_norm] | @tsv' \
   exp/<run>/logs/train.jsonl
# did the round help
jq -r 'select(.kind=="eval") | [.round, .source, .selected, .spearman] | @tsv' \
   exp/<run>/logs/eval.jsonl
```

Three things about what is logged are worth knowing before reading a curve:

- **Per batch, tagged by source** (`batch/loss` and `batch_care/loss`,
  `batch_navtrain/loss`). The split is the reason this exists. A CARE batch
  produces no imitation term and has no drivable-area column, so the two halves of
  an epoch are not measuring the same thing and their mean is not a quantity.
- **`trajectory_loss` is a navtrain-only average.** CARE rows never produce the
  term at all, so `epoch/trajectory_loss` is a mean over roughly half the epoch's
  batches, not over all of them.
- **`grad_norm` is the total norm before clipping**, taken after
  `scaler.unscale_`, so it is in true units and comparable across steps. It goes
  non-finite on a step the AMP scaler is about to skip, which is information rather
  than a bug; those points are `null` in the JSONL and absent from the event file.

The step axis is global across rounds and survives the process boundary --
`tracking.py` resumes it from the tail of the JSONL -- so a run resumed at round 3
continues one curve instead of starting a second one at zero.

`--no-tensorboard` on `train.py`, `loop.py` or `eval.py` drops the event files and
keeps the JSONL. It exists because importing `torch.utils.tensorboard` pulls
TensorFlow in this environment and costs a few seconds of startup; nothing per step.

## Files

| file | role |
|---|---|
| `config.py` | paths, round schedule, loss weights -- one dataclass |
| `scoring.py` | true sub-scores: a nuPlan backend and a map-free CARE one |
| `care.py` | CARE clip pickles -> scenes the model and the scorer can read |
| `data.py` | both sources behind one interface, plus batching |
| `model.py` | RAP wrapped so arbitrary trajectories can be scored |
| `buffer.py` | the append-only buffer of scored trajectories |
| `collect.py` | one round: sample, score, append |
| `train.py` | one round: fit the scorer to the whole buffer |
| `loop.py` | collect -> train -> repeat, resumable |
| `eval.py` | selected / pool_best / rank correlation on held-out scenes |
| `tracking.py` | per-batch scalars to JSONL and TensorBoard event files |
| `export.py` | round checkpoint (a delta) -> a self-contained one |
| `test_eval_sampling.py` | invariants of which proposals `eval.py` truly scores |

## Notes

- **Never parallelise the scorer with threads.** `compute_navsim_score` builds a
  module-level `PDMSimulator`/`PDMScorer` pair and `PDMScorer` keeps per-call state
  on `self`. Concurrent calls in one process interleave and corrupt each other's
  results with no exception -- wrong labels, silently, which is the one failure
  nothing downstream can detect. `score_workers` is a process count.
- **Every pool here uses `spawn`.** Scoring runs after the model has been on the
  GPU, so the parent holds an initialised CUDA context, and forking one is unsafe:
  observed in this project as a pool sitting at 0% CPU forever, holding the
  parent's image, having scored nothing.
- **A batch is never mixed across sources.** navtrain samples carry 4 cameras and
  CARE windows carry 1, and those tensors do not stack.
  `data.iterate_batches` interleaves per-source loaders instead.
- **`camera_feature`, not `rendered_camera_feature`.** `RAPConfig.distill_feature`
  is `False`, so `AgentLightningModule._step` runs and the swap to the rasterized
  view in `_step_distill` does not. The checkpoint was trained on photographs, and
  a CARE clip's photographs are the extracted video frames.
- **The buffer stores trajectories and scores, never features.** The refiners
  train, so a round-1 trajectory has to be re-featurised by the round-3 model
  before round 3 can learn from it. That is also why the ViT stays frozen: it is
  the one part that could be cached, and freezing it means the round that scored a
  trajectory and the round that trains on it agree about what the scene looks like.
- **CARE boxes have no temporal association.** `track_tokens` and
  `instance_tokens` are re-hashed every frame -- consecutive frames share zero
  tokens of either kind. The collision test is per-timestep and does not need
  association, but it does mean a box that flickers out for one frame opens a
  one-timestep hole in the obstacle.
- **CARE ego poses come from OpenVO odometry** and are noisy enough that the
  human trajectory's own acceleration profile can exceed nuPlan's comfort
  thresholds. Comfort is measured on absolute thresholds rather than relative to
  the human for a different reason (a crashing human makes a relative bar
  meaningless), but both are worth remembering when reading a CARE comfort column.
- **`scripts/` is outside this package.** `scripts/training/run_rl_quicktest.sh`
  is the one wrapper: round 1 (i.e. from the pretrained planner) at 5 epochs on
  the full CARE export with navtrain mixed in, then `eval.py --baseline`. It asks
  for **one** GPU, because nothing in `rl/` is distributed -- no Lightning, no
  DDP, so the supervised script's `SLURM_JOB_NAME=bash` and `WANDB_MODE` lines
  have nothing to act on and are deliberately absent. Its `--cpus-per-task` is
  sized for `score_workers`, which is a process count. Everything else is the
  `python rl/...` commands above.
