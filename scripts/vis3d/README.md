# vis3d

Turns a driving video into 3D boxes, lane dividers and traffic-light states,
rasterized back over the frames.

Home directory example:

```bash
BASE=/fs/nexus-projects/sim2real/aliu/RAP
```

## End to end

Two commands, in two different environments. Everything below is either setting
them up or re-running a part of the first one.

```bash
mkdir -p $BASE/data/CARE_YTB          # both steps error on a missing dataset dir

conda activate vis3d
$BASE/scripts/vis3d/run_render_dataset.sh          # fetch, extract, render; one job per clip
# wait for the jobs to finish

conda activate rap
python $BASE/scripts/vis3d/generate_ras_logs.py    # runs -> navsim logs under $BASE/CARE
```

Neither takes arguments in the common case: both are configured by editing the
block at the top of the file, and `DRY_RUN=1` / `--dry-run` report what each
would do without touching anything.

Two things that are easy to get wrong: run `run_render_dataset.sh` **directly,
not through sbatch** -- it only submits, and exits in a second. And run the
export in `rap`, not `vis3d`: it writes pickles the training env has to
unpickle, so it belongs in the env that will read them.

## Setup

All four stages run in **one conda env, `vis3d`** (py3.11 / torch 2.2.0 /
cu121). That is what the pins and comments in `requirements.txt` exist to hold
together: stage 1 (GroundingDINO) used to need its own py3.8 / torch-1.12 env,
stage 2 (UniDepth) a `unidepth` env, and the stage-4 plot borrowed `drivoR`.
Each note below is what it took to bring one of those onto the shared floor.
`run_render_vis3d.sh` activates `vis3d` itself, so no stage needs a per-stage
activation and nothing here switches envs mid-pipeline.

Three things `pip install -r requirements.txt` does *not* cover, each with its
own step below: the Grounded-Segment-Anything checkout and its patch, UniDepth
(installed `--no-deps`), and the checkpoints. OpenVO, the learned stage-4 ego
motion model, is **deprecated**; its two source builds are kept at the
[bottom of this file](#deprecated-openvo) for anyone who still needs it.

Ego-speed estimation (`run_odometry.sh`, section 3c) also uses a second env,
**`processor`**, for one step: `estimate_intrinsics.py` (WildCamera needs
`mmcv.cnn`, which `vis3d` does not have). The script switches to it on its own.

The export at the end is the one part that does not run here at all -- it needs
the training env, `rap`. See step 6.

### 0. The vis3d tree

Every stage script lives in `$BASE/vis3d/`, along with two vendored trees that
have no upstream of their own (`openvo/`, `PCADetection/`). **It is untracked by
this repository and not in `.gitignore`**, so a fresh clone does not bring it and
only `Grounded-Segment-Anything/` (step 2) can be rebuilt from scratch. Copy it
first:

```bash
cp -a <existing>/RAP/vis3d $BASE/vis3d
```

### 1. The env

```bash
conda create -n vis3d python=3.11
conda activate vis3d
conda install -c conda-forge ffmpeg     # yt-dlp's muxer and frames_to_video.py
conda install -c conda-forge nodejs     # yt-dlp's JS runtime, see below
pip install -r $BASE/scripts/vis3d/requirements.txt
```

`nodejs` is easy to miss: it is not a Python package, so it cannot be in
`requirements.txt`, and without it YouTube extraction warns and silently drops
formats until the fetch fails. Reddit links do not need it. Any node on `PATH`
works, but putting it in the env is what makes `conda activate vis3d` enough.

`requirements.txt` carries the reasoning for every pin that is not simply
"latest" -- read it before bumping anything, particularly torch, numpy,
xformers, transformers and timm, which are load-bearing for one stage each.

### 2. Stage 1: Grounded-Segment-Anything

Clone the detector and fetch the SAM checkpoint (2.4 GB):

```bash
cd $BASE/vis3d
git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git
cd Grounded-Segment-Anything && git checkout 126abe6
curl -L -o sam_vit_h_4b8939.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
git apply $BASE/scripts/vis3d/groundingdino-no-cuda-op.patch
```

That last patch is required, not optional -- see below. It is a two-hunk edit to
`ms_deform_attn.py` and re-cloning without it puts stage 1 back on a torch the
rest of this env cannot have.

Nothing needs compiling. GroundingDINO's `_C` CUDA op is not built -- its
source includes `<THC/THCAtomics.cuh>`, a header PyTorch deleted after 1.12, so
it cannot compile against a modern torch. Upstream's `ms_deform_attn.py` still
dispatches to that op whenever a CUDA tensor shows up, which raises a
`NameError` once the import has failed, so `groundingdino-no-cuda-op.patch`
makes the dispatch conditional on the op actually being present. It then falls
back to GroundingDINO's own pure-PyTorch deformable attention, which still runs
on the GPU and reproduces the compiled op's boxes and scores to six decimals.
That fallback is what lets stage 1 share this env rather than needing the old
py3.8/torch-1.12 one.

### 3. Stage 2: UniDepth

Install it separately, with `--no-deps`:

```bash
pip install --no-deps \
  "unidepth @ git+https://github.com/lpiccinelli-eth/UniDepth.git@9d74fd906ca11cf055a83473607cb43330f0ac6c"
```

`--no-deps` is not an optimisation. UniDepth's `setup.py` declares an entire
`pip freeze` as `install_requires` -- `botocore==1.34.54`, `black`, `flake8`,
`torch==2.2.0`, `certifi==2022.12.7` and forty more -- most of which the code
never imports. Its real imports are already in `requirements.txt`. Expect
`pip check` to keep reporting those undeclared pins afterwards; that is the
known, intended state of this env.

### 4. Stage 4: ego motion

Nothing to install. Stage 4 runs the `pointcloud` method -- numpy, OpenCV and
pycocotools, already in `requirements.txt` -- on stage 2's point maps.
`run_render_dataset.sh` sets `VO_METHOD=pointcloud` itself.

For metric ego *speed*, prefer section 3c (`run_odometry.sh`): on 18 of the 50
CARE clips stage 2's depth collapses (the ego's own bonnet comes back 25-30 m
away), and every depth-based ego-motion method inherits that.

The learned OpenVO method is deprecated; see [Deprecated: OpenVO](#deprecated-openvo).

### 5. Checkpoints

| file | stage | how |
| --- | --- | --- |
| `vis3d/Grounded-Segment-Anything/sam_vit_h_4b8939.pth` | 1 | the `curl` in step 2 |
| GroundingDINO `.pth` | 1 | pulled from HuggingFace on first run |
| `weights/yolopv2.pt` | 1b | the `curl` below |

```bash
curl -L -o $BASE/weights/yolopv2.pt \
  https://github.com/CAIC-AD/YOLOPv2/releases/download/V0.0.1/yolopv2.pt
```

### 6. The training env, for the export

`generate_ras_logs.py` needs nothing but numpy, PIL and the stdlib, so there is
nothing to install -- `rap` already has all three. Run it there rather than in
`vis3d` because it writes pickles the training job unpickles, and the two envs
are far apart (py3.11 / numpy 1.26 against py3.9 / numpy 1.23). A pickle written
by a numpy 2.x env dies in a 1.x one with `No module named 'numpy._core'`, at
log-load time inside the training run, and nothing checks.

```bash
conda activate rap
python -c "import numpy, PIL; print(numpy.__version__, PIL.__version__)"
```

### Verifying the env

```bash
conda activate vis3d
cd $BASE/vis3d                       # the GSA paths below are cwd-relative

python -c "import torch, xformers, timm, unidepth; print(torch.__version__)"
python -c "import sys; sys.path += ['Grounded-Segment-Anything',
                                    'Grounded-Segment-Anything/GroundingDINO']
from GroundingDINO.groundingdino.models import build_model; print('stage 1 ok')"
```

Expected: `2.2.0+cu121`, then `stage 1 ok`. Stage 1 also prints
`groundingdino._C not available; using the pure-PyTorch multi-scale deformable
attention` -- that warning is the patch working, not a problem. None of this
needs a GPU, so it runs on the login node.

A clean import of all of them from one interpreter is the property this env
exists to have.

## 1. Ingest a clip

Set `NAME` and `YTB_LINK` in `process_ytb.sh`, then:

```bash
bash $BASE/scripts/vis3d/process_ytb.sh
```

Writes `data/YTB/<clip>/<clip>.mp4` and `frames/000000.jpg …` at 2 Hz, every
frame fitted to 1920x1080 by a uniform scale plus a crop of the overflow
(`--size`/`--crop-bias`; never a stretch, since that would make fx/fy disagree
with the real lens). `run_render_dataset.sh` does the same for a batch.

### Cookies

If yt-dlp fails with `Sign in to confirm you're not a bot` or a 403, refresh
`scripts/vis3d/cookies.txt`:

1. Install a cookie exporter — e.g. **Get cookies.txt LOCALLY**
   (Chrome Web Store), which writes Netscape-format files.
2. Open `youtube.com` in Chrome, signed in.
3. Click the extension → **Export** / **Copy**.
4. Paste into `$BASE/scripts/vis3d/cookies.txt`

### Frame size

Every clip must be at 1920x1080 -- navsim's own CAM_F0 size -- because the
intrinsics, the lifted boxes and the export all assume it. `process_ytb.py`
has fitted frames to `--size` since the flag existed, and `run_render_dataset.sh`
passes `TARGET_SIZE` to it, so anything ingested by the current pipeline is
already right. Clips extracted before that are whatever their source video was.

`refit_frames.sh` finds and repairs those:

```bash
DRY_RUN=1 ./refit_frames.sh          # what differs from TARGET_SIZE, changing nothing
./refit_frames.sh                    # repair them and submit the re-renders
./refit_frames.sh changelane         # just this clip
RENDER=0 ./refit_frames.sh           # refit the frames, do not re-render
```

A clip at the target size is left alone, so this is safe to run over the whole
dataset whenever the target changes. One that is not cannot be repaired in
place -- the run directory is a function of the frames it was rendered from, so
every file in it is wrong the moment the frames change. Both are moved aside
intact instead, `frames/` to `frames_old/` and `$RUN/` to `old/`, and a new
`frames/` is extracted from the clip's own mp4 at the right size before the
render is resubmitted to rebuild `$RUN`. Nothing is deleted, and a second refit
parks its predecessors at `frames_old_2/` and `old_2/` rather than overwriting
the first.

Two things it handles that are easy to get wrong by hand:

- **Hand-drawn boxes are in pixels**, so they mean something different on a new
  grid. `manual_boxes.json` is carried into the rebuilt run through the same
  transform the pixels went through, rather than left to land 1.5x off.
- **`SOURCE_HZ` must be the rate the clip was originally extracted at.** Wrong,
  and the clip is resampled in time rather than resized, which shifts every
  frame index. The refit compares its new frame count against the old one and
  says so loudly if they differ.

Note that a refit only makes the pixel grid consistent; upscaling a 720p source
to 1080p adds no detail it did not have.

## 2. First pass

Set the CONFIG block in `run_render_vis3d.sh`:

```bash
VIDEO="beepbeep"
RUN="1"                  # output subdir; "" = clip dir itself
RUN_STAGE1_MASKS=1       # object masks   -> mask_results_preds.json
RUN_STAGE1B_LANES=1      # lane dividers  -> lane_masks/
RUN_STAGE2_DEPTH=1       # point maps     -> samples-pseudodepth/
RUN_STAGE3_LIFT=1        # lift + raster  -> boxes_3d.json, vis3d/, vis3d_overlay/
RUN_STAGE4_VO=1          # ego motion     -> ego_poses.txt, ego_trajectory.png
EXPORT_NAVSIM=1          # navsim log     -> navsim_logs/, sensor_blobs/
LANE_THRESHOLD=0.5       # lane-line probability cut
```

One script runs the whole pipeline: stages 1-3 render the clip, stage 4
estimates the ego trajectory, and the export writes the navsim-format log from
both. Every part is independently switchable, so a later pass can re-run just
the half it needs.

```bash
sbatch $BASE/scripts/vis3d/run_render_vis3d.sh
```

Every CONFIG entry above also reads from the environment, which is what lets a
whole dataset be run without editing the file. `run_render_dataset.sh` submits
one job per clip in `data/$DATASET` -- separate jobs rather than a loop inside
one, so a clip that fails takes only itself down:

```bash
DATASET=CARE_YTB $BASE/scripts/vis3d/run_render_dataset.sh          # all clips
DATASET=CARE_YTB $BASE/scripts/vis3d/run_render_dataset.sh blocker  # named clips
DRY_RUN=1 ... run_render_dataset.sh    # print the sbatch lines, submit nothing
LOCAL=1 ...    run_render_dataset.sh   # run here instead of submitting
FORCE=1 ...    run_render_dataset.sh   # re-run clips that already have boxes_3d.json
```

Clips whose delivered run already holds `boxes_3d.json`, and clips with no
extracted frames, are skipped rather than re-run, so the same command can be
issued again after adding a clip to the dataset.

### Processing at 10 Hz, delivering at 2 Hz

New clips are extracted at **10 Hz**, run through every stage at 10 Hz, and then
subsampled to the 2 Hz clip that is actually delivered. Everything in the
pipeline that reasons across time gets easier as the frames get closer together
and none of it gets harder: association gates on how far a box's projected
centre moves between frames, the yaw and extent medians get five times the
samples over the same stretch of road, and UniDepth's metric scale drifts with
elapsed time rather than jittering per frame, so each track link spans a fifth
as much of it (measured on `buick_nearmiss`: frame-to-frame scale spread 1.23x
at 0.1 s against 1.99x at 0.5 s). It costs 5x the stage-1 and stage-2 compute
and 5x the point maps on disk.

    <clip>/frames_10hz/    what the pipeline runs on
    <clip>/10hz/           the run at 10 Hz
    <clip>/frames_2hz/     every 5th frame, renumbered (symlinks)
    <clip>/2hz/            the delivered clip; frames/ and 1/ are linked to
                           these two afterwards if those names are free

`SUBSAMPLE_STRIDE=1 SOURCE_HZ=2` restores the old behaviour exactly.

The smoothing knobs are all frame *counts* whose intent is a duration, so their
defaults are scaled by `SUBSAMPLE_STRIDE` -- at stride 5, `MIN_TRACK_LEN` 2 -> 10
and `MAX_FILL` 1 -> 5, both still one second and half a second. Override any of
them and your number is used unscaled. `MAX_PX` is floored at 100 rather than
scaled straight to 50: real motion shrinks with the interval but the jitter from
noisy box depth does not, so a strictly scaled gate is tighter than the noise it
has to tolerate.

One thing to watch: `MIN_TRACK_LEN` at stride 5 asks for a full second of track,
and on a sparse clip that is a lot. Retrofitting the existing set with
`MIN_BOX_SCORE=0.45` as well took `deercross` from 31 boxes to 3. The score floor
defaults to 0 for new clips, which is the safer half of that pair.

## 2b. Correcting boxes the detector missed

GroundingDINO fails outright on some footage rather than degrading: `closetruck`
is 4 detections across 60 frames, its truck backlit into blown-out white and half
out of the left edge. `vis3d/annotate_boxes.py` serves a small annotator over a
clip's frames so those objects can be drawn by hand.

```bash
cd vis3d
python annotate_boxes.py --clip closetruck --dataset CARE_YTB --run 1
```

It binds `127.0.0.1:8791` and prints the url. Under VS Code Remote the port is
forwarded for you; over plain ssh, forward it yourself:

```bash
ssh -N -L 8791:127.0.0.1:8791 <host>
```

A track is either an **object** you are adding or a **reject region** that
deletes. `+ new track` (`n`) makes the first; `+ reject region` (`r`) makes the
second, drawn in red. A reject region suppresses every detector box it covers
and contributes none of its own -- it is how a false positive is removed rather
than replaced, which drawing over cannot do, since a drawn box swaps yours in
for theirs. The recurring left-edge detection on `buick_nearmiss` (a car found
on the ego bonnet across 10 frames) is the case it exists for.

Boxes are **keyframed**, not drawn per frame: a track carries a box on the frames
you draw on and is interpolated linearly between them, so a truck crossing the
frame is two or three drags rather than sixty. Nothing is extrapolated before a
track's first keyframe or after its last, and `mark gone` ends a track's run at a
frame it leaves. The detector's own boxes are drawn greyed out underneath, which
is how you see at a glance which frames it actually covered.

Saving writes `manual_boxes.json` into the run directory. That file is the source
of truth for hand-drawn boxes and nothing in the pipeline overwrites it.

`apply_manual_boxes.py` folds it into stage 1's output, prompting SAM with each
box to get a real mask (`--no-sam` uses the box rectangle, no GPU):

```bash
python apply_manual_boxes.py --output_dir ../data/CARE_YTB/closetruck/1
```

A manual box lands as a mask entry in `mask_results_preds.json` at score 1.0 with
`"manual": true`, so stage 3 lifts it, stage 3b smooths it and the navsim export
carries it out with no knowledge that a person drew it. A detector box overlapping
a manual one by `--iou` (default 0.5) is dropped, so correcting a bad detection
means drawing the right box over it.

A reject region drops the same boxes by the same `--iou` but adds no entry, and
the two are counted separately in the merge's output -- a region that deletes
nothing is reported, since that means it is not over what it was meant to
delete. Where a detection is covered by both, the reject wins: a drawn box says
"it is really here", a reject says "nothing here is real", and only the second is
ever drawn by accident. A reject also blocks adoption over what it covers, so a
rejected box cannot come back through a neighbouring track's identity.

On the frames a drawn track does *not* cover, the opposite happens: a detection
overlapping where the track just was is **adopted** into it -- it keeps its own
mask and score and gains the track id, and is relabelled to the track's class.
This is what makes hand annotation *continue* a detection rather than sit beside
it. A detector that catches an object in one- and two-frame glimpses produces
tracks that `MIN_TRACK_LEN` deletes as flicker, and drawing boxes over the frames
it missed does not save them on its own: drawn and detected boxes never merge in
stage 3b's associator, deliberately, because splicing them there means guessing
an identity from a lifted position that is itself a guess. Matching in image
space, before the lift, needs no such guess. `--adopt_iou` (default 0.3) is the
overlap that claims a detection and `--adopt_gap` (default 5) how many frames a
track may go unseen first; `--adopt_gap 0` turns it off. A null keyframe drops
the track's tip, so marking an object absent is not overruled by a detection.

The merge is idempotent: the detector's untouched output is copied aside to
`mask_results_preds.detector.json` on the first run and every later run re-reads
*that*. Edit the boxes, re-run, and the result is the same as if it had merged
once. `run_render_vis3d.sh` runs this automatically as stage 1c whenever the run
directory holds a `manual_boxes.json` (`APPLY_MANUAL_BOXES=0` to skip), so a clip
corrected once stays corrected across every later re-render.

Two things worth knowing:

- Re-running **stage 1** overwrites `mask_results_preds.json` but never
  `manual_boxes.json`. Whether that file is a fresh detection or a previous merge
  is decided by whether it carries `"manual"` entries, so stage 1c re-merges your
  boxes onto the *new* detections and refreshes the pristine copy from them.
- `MIN_TRACK_LEN` in `run_render_vis3d.sh` (default 3) drops tracks seen in fewer
  than three frames, but **drawn tracks are exempt** -- one frame of a box someone
  drew is still a deliberate frame -- and so are the detections adopted into them.
  Stage 3b now lists what it dropped by frame and class rather than only counting
  it, which is the line to read when a box you can see in `vis/` is missing from
  `vis3d_overlay/`.

## 2c. Correcting the lifted 3D boxes

`annotate_boxes.py` corrects what the *detector* saw, in 2D, upstream of the
lift. That cannot reach an error the lift itself made -- a box on a real vehicle
at the wrong depth, or the ego's own bonnet lifted into a 4 m car sitting 11 m
ahead. `--boxes3d` edits the lifted boxes directly:

```bash
cd vis3d
python annotate_boxes.py --clip opposing_crash --run 2hz --boxes3d
```

Camera view on the left with the boxes projected onto the frame, bird's-eye on
the right. Both matter: a box can look right in projection and be at completely
the wrong range, and only the BEV shows that. Drag in the BEV to move, `WASD`
to nudge, `QE` to yaw, `1`-`6` to resize, `N` to add, `Del` to delete.

**It edits the 2 Hz run.** That is what the navsim export reads and what
training sees, so a correction there is final; and it is a fifth of the frames,
which for hand work is the difference between a tool being used and not. The
10 Hz run is an intermediate -- correcting it would mean the edit is re-derived
through subsampling rather than authoritative, and a 2 Hz correction cannot be
pushed back to 10 Hz without inventing the four frames it says nothing about.

Saving writes `manual_boxes_3d.json` beside `boxes_3d.json`. Apply it with:

```bash
python apply_manual_boxes_3d.py --output_dir ../data/CARE_YTB/opposing_crash/2hz
cd visualization && python raster_frames.py --output_dir <run> --frames_dir <frames_2hz>
```

The edits live in their own file rather than being written into `boxes_3d.json`
once, for the same reason `manual_boxes.json` does: stage 3 rewrites
`boxes_3d.json` from scratch, so an edit made in place survives until the next
re-lift and then vanishes without saying so. The first apply copies the lift's
own output to `boxes_3d.lifted.json` and every later one re-reads that, so
applying twice is the same as applying once.

`delete` and `replace` are anchored to a *position*, not an index -- an index
means nothing across a re-lift, which rebuilds the boxes in a different order
and in slightly different places. An anchor that no longer matches anything is
reported rather than skipped in silence, because that is exactly the signal that
the lift has moved under your annotation.

## 2d. Reviewing the boxes with Gemini

`annotate_boxes.py --boxes3d` fixes a box you have already found. Finding them
is the other half, and on 50 clips of 155 frames it is the half that does not
finish. `vis3d/gemini_review_boxes.py` renders each sampled frame with its boxes
as numbered wireframes, asks Gemini which ones are wrong, and writes what comes
back into the same `manual_boxes_3d.json` that the annotator writes:

```bash
export GEMINI_API_KEY=...
./run_gemini_review.sh                                     # the CONFIG block
VIDEO=deer_family DATASET=test FIND=deer ./run_gemini_review.sh
VIDEO=closetruck NOTE="frames 40-90: the box on the white pickup sits left of it" \
    ./run_gemini_review.sh
DRY_RUN=1 ALL=1 ./run_gemini_review.sh                     # the whole dataset, no writes
```

It writes nothing but that file and a `gemini_review/` directory holding
`report.md`, `findings.json`, and the reviewed frames with the corrections drawn
on them -- white for what was accepted, grey for what was dropped and why. Read
those, delete the edits you disagree with, then `apply_manual_boxes_3d.py` as in
2c. The review is a proposal; applying it is still your call.

### What it is asked, and what it is not

Only what the picture shows: is the wireframe on a real road user, does it cover
that road user, does its thick front face point the way the object does, is
there something with no box on it. It is never shown a box's size or range in
metres and never asked to judge them. On these clips UniDepth's guessed focal
length runs several times long, so every lifted box is metrically several times
small -- a "car" 2.5 m long, and on `deer_family` closer to 0.9 m -- and the
error cancels on reprojection, which is why the overlays look right anyway. A
reviewer shown those numbers would flag every frame of every clip and be right
about nothing that can be fixed downstream of the lift.

That same cancellation is what makes an image-space correction the right thing
to ask for. Gemini returns the 2D box the wireframe *should* have projected to,
and `fit_box_to_target` solves for the translation and scale that put it there:
translation fronto-parallel, so the range the ground calibration fixed is left
alone, and scale about the box's bottom face, so a resize cannot lift a vehicle
off the road. Nothing here asks a language model for a metre.

### Why a stride is affordable

`--stride 5` reviews 31 frames of 155, and then each correction is pushed along
its object's track -- associated with `smooth_boxes.associate`, the same
association the smoother already trusts. A false positive found once is deleted
everywhere it appears; a misplaced box is corrected on every frame of its track,
with the offset carried in *pixels* and converted back to metres against each
frame's own range, because a mask that sits off its object stays off it by the
same amount on screen while the metres grow as the object recedes. A track
reviewed at several frames is split at the midpoints. `PROPAGATE=frame` turns
all of it off and edits only the frames actually looked at.

### Telling it what you already found

`NOTE` is put in front of the reviewer as ground truth, so a fault you spotted
is confirmed and turned into an edit rather than re-discovered. A note may name
its frames -- `"frames 40-90: ..."`, `"frame 73: ..."` -- and those frames are
then reviewed whether or not the stride sampled them. `NOTES_FILE` holds one per
line and may group them under `[clip]` headings, so one file can carry a review
pass over the whole dataset.

`FIND` asks for a class outright: `FIND=deer` on a clip whose detector only ever
found cars boxes every deer in the frame. An added box for a class with no
example in the clip is seeded from that class's real-world *proportions*
rescaled by what the clip's own cars say a metre is here -- the shape from the
prior, the scale from the clip, since the two disagree by the factor above.
Adds are keyframes: where the same object is added on consecutive reviewed
frames, the frames between them are interpolated, the same rule
`apply_manual_boxes.py` uses for hand-drawn tracks. Nothing is extrapolated past
the first or last frame someone actually looked at.

### Rate limits

The free API tier allows 5 requests a minute, and above it every extra request
is a 429 that still spends a request. `RPM` holds the whole run under a ceiling
and `WORKERS` should be 1 under a low one -- parallelism buys nothing you are
not allowed to spend. A 429's own retry hint is honoured, so a run that hits one
waits the 40-odd seconds it is told to rather than a backoff guessed here.

## 3. Re-render on CPU

`rerun_vis3d.sh` is the short way to do this: set `DATASET`/`VIDEO`/`RUN` and
the six stage flags at the top of the file, run it directly, and it submits the
job. Everything else still comes from `run_render_vis3d.sh`'s CONFIG block.

```bash
./rerun_vis3d.sh
```

The flags reach the job through `sbatch --export=ALL,VAR=...` rather than a
`VAR=... sbatch` prefix, so an assignment cannot go missing on the way and
leave a stage running against the default run directory.

Or set the flags in `run_render_vis3d.sh` directly:

Reuse the first pass and re-run only lift + rasterize:

```bash
RUN="2"
RUN_STAGE1_MASKS=0
RUN_STAGE1B_LANES=0
RUN_STAGE2_DEPTH=0
RUN_STAGE3_LIFT=1        # the one stage being re-run
RUN_STAGE4_VO=0
EXPORT_NAVSIM=0
REUSE_MASKS_FROM="1"     # "" = borrow from the clip dir
REUSE_LANES_FROM="1"
REUSE_DEPTH_FROM="1"
OVERLAY_ALPHA=0.6        # raster opacity, free to re-tune
```

```bash
sbatch -J <job_name> $BASE/scripts/vis3d/run_render_vis3d.sh
```

Reused artifacts are symlinked in. Never enable a stage whose output is linked —
writes follow symlinks into the run you borrowed from.

A `REUSE_*_FROM` left `""` while the artifact already sits in this run's own
directory is a no-op, not an error: that is the shape of re-running only the
ego-motion half over a render that already landed here.

## 3b. Ego motion only

Once the render half is done, the trajectory and the log can be redone alone —
re-tuning `VO_METHOD`, or picking the export back up after a failed VO run:

```bash
RUN_STAGE1_MASKS=0
RUN_STAGE1B_LANES=0
RUN_STAGE2_DEPTH=0
RUN_STAGE3_LIFT=0
RUN_STAGE4_VO=1
VO_METHOD="pointcloud"   # CPU, from stage 2's point maps. openvo is deprecated -- see the bottom
EXPORT_NAVSIM=1
SOURCE_HZ=2              # must match process_ytb.py --hz
```

Set `VO_METHOD` explicitly when running `run_render_vis3d.sh` directly: its own
default is still `openvo`. (`run_render_dataset.sh` sets `pointcloud` for you.)

`ego_trajectory.png` is written on every VO run and is the check worth making:
a trajectory can look healthy on average speed while its path is a random walk.
It is drawn from `ego_poses.txt` -- these point-cloud poses -- and **not** from
the dash/road speeds of section 3c, so re-running odometry does not change it.

## 3c. Ego speed: `run_odometry.sh`

Metric ego speed for every clip in a dataset, measured from the road itself
rather than from a depth network. It exists because depth-based ego motion
(stage 4, and OpenVO before it) is badly wrong on this footage: on 18 of the 50
CARE clips the path comes out 20-100x too short, and OpenVO's per-frame speed on
`changelane` has essentially no correlation with the dashcam's own speed readout
(r = 0.07).

### How it works

1. **Road plane.** Camera tilt and height above the road, so a road pixel maps to
   metres ahead. Measured from two parallel lane lines and the lane width when
   possible (`calibrate_plane.py`); otherwise fitted from the clip's car
   detections -- a car's height in pixels grows linearly with how low it sits in
   the frame, and one line fit gives the horizon and the camera height
   (`vehicle_horizon.py`). `road_plane.json` records which, as `quality`.
2. **Dash odometry** (`detect_dashes.py`, `dash_odometry.py`), for clips with
   dashed lane lines: track individual dashes between frames. The scale comes
   from the known dash spacing for the clip's `country` in `info.json` (US
   12.2 m, RU 16.3 m, ...), so an imperfect plane largely cancels out.
3. **Road odometry** (`road_odometry.py`), for everything else: warp each frame
   to a bird's-eye view of the road and find the forward shift that lines it up
   with the next. Only edges crossing the road count (lane lines run with the
   motion and carry no speed); painted markings -- arrows, stop bars, crosswalks,
   dash ends -- are weighted up, and dark pixels such as moving shadows are
   left out. A step is kept only when the 1-, 2- and 3-frame matches agree.
4. **Selection** (`select_odometry.py`): dash if the road plane was calibrated
   from lane lines, otherwise road. Written to `ego_speed.json`.

### Running it

Submit through sbatch, with settings in the environment and `--export=ALL`:

```bash
cd $BASE/scripts/vis3d

# every clip in a dataset, every stage
DATASET=CARE_YTB sbatch -J odometry --export=ALL run_odometry.sh

# re-do road odometry + selection + report after changing road_odometry.py
DATASET=CARE_YTB STAGES=road,select,report FORCE=1 sbatch -J odometry --export=ALL run_odometry.sh

# refresh only the selection and the report (seconds)
DATASET=CARE_YTB STAGES=select,report sbatch -J odometry --export=ALL run_odometry.sh

# one clip (the report still covers the whole dataset)
DATASET=CARE_YTB CLIPS=back_up STAGES=road,select,report sbatch -J odometry_back_up --export=ALL run_odometry.sh

# interactively on the login node, CPU-only stages
DATASET=CARE_YTB CLIPS=four_way STAGES=road,select bash run_odometry.sh
```

Two ways a submission goes wrong without an error:

- **The dataset does not reach the job.** Without `DATASET` in the job's
  environment it silently runs on the default, `data/test`. Check the first line
  of the log (`my_dump/<job name>.out.<job id>`): it must say
  `=== dataset : .../data/<your dataset>`.
- **Comma lists inside `--export` are split.** `--export=ALL,STAGES=road,select`
  delivers `STAGES=road` and drops `select`. Put `STAGES` in the environment as
  above; the log's `=== stages :` line shows what arrived.

### Settings

| variable | default | meaning |
| --- | --- | --- |
| `DATASET` | `test` | directory under `data/` |
| `CLIPS` | every clip with `frames/` | space-separated clip names |
| `STAGES` | `lanes,intrinsics,vo,plane,dash,road,select,report` | comma-separated; see below |
| `FORCE` | off | `1` regenerates each selected stage's own output even where it exists |
| `HZ` | `auto` | frame rate; a number forces it for every clip |
| `RUN` | `1` | run directory inside each clip |
| `PARALLEL` | half the CPUs | clips processed at once |
| `NO_HEADING` | off | `1` skips the slow heading pass in the ground-truth comparison plot |
| `PITCH_<clip>`, `HEIGHT_<clip>` | unset | hand-set road plane for one clip |

### Stages

| stage | writes | notes |
| --- | --- | --- |
| `lanes` | `lane_masks/` | `detect_lanes.py` |
| `intrinsics` | `camera_intrinsics.json` | `estimate_intrinsics.py`; GPU, `processor` env |
| `depth` | `samples-pseudodepth/` | UniDepth; slow, not in the default list |
| `vo` | `ego_poses.txt`, `ego_trajectory.png` | point-cloud VO; needs `samples-pseudodepth/` |
| `plane` | `road_plane.json` | lane lines, else car detections, else vanishing point / nominal |
| `dash` | `dashes.json`, `dash_detections.jpg`, `dash_speed.json`, `dash_odometry_vs_ground_truth.png` | the plot only for clips with ground truth |
| `road` | `road_speed.json` | skipped where it exists unless `FORCE=1` |
| `select` | `ego_speed.json` | always re-run; JSON only |
| `report` | `data/<dataset>/odometry_report.json` + a table in the log | always the whole dataset |

A stage **generates the inputs it needs** when they are missing, instead of
skipping: `lane_masks/` and `camera_intrinsics.json` for the plane,
`camera_intrinsics.json` for road odometry, `road_plane.json` for dash odometry
and selection, and a road speed when selection has nothing to choose from.
`FORCE` never regenerates these borrowed inputs, only the stage's own output.
Two inputs it cannot make, and names in the log when missing:
`mask_results_preds.json` and `drivable_masks/`, both from
`run_render_vis3d.sh` stage 1. Without them road odometry cannot fit the car
plane, mask vehicles, or find the road and the bonnet edge.

### Frame rate

Every speed is metres per frame times frames per second, and both matchers
search a fixed distance per frame. With `HZ=auto` each clip's rate is taken from
`"hz"` in its `info.json` if present, otherwise by matching its first extracted
frames against its own mp4 at candidate rates (so a clip trimmed at the end
still reads correctly), otherwise 10 Hz with a warning. The log prints the rate
and how it was decided.

**Both estimators were validated only at 10 Hz.** Any other rate is flagged as
unchecked. `back_up` is 2 Hz: road odometry matched none of its 121 frames,
which gives the right total for that stationary clip but no measured speed.

### Reading the results

`ego_speed.json` holds one speed per frame (`null` where nothing was measured),
plus `source` (dash or road), the reason for the choice, `plane_quality` and
`reliability: "unverified"` -- nothing yet tells, without ground truth, whether
a chosen speed is right.

Ground truth comes from **11 clips that burn their speed into the frame**,
collected in `data/test/osd_ground_truth.json`. On those, road odometry covers
this share of the true distance (gaps between measured frames filled, scored
up to the last on-screen reading):

| within ~35% | rough | fails |
| --- | --- | --- |
| `turn_blocker` 101%, `turn_overtake` 106%, `yield_runway` 109%, `ambulance` 119%, `close_bike` 127%, `reserved_lane` 133% | `four_way` 150%, `exit_now` 54% | `changelane` 31%, `too_close` 44% (highway speed); `close_slam` 34% (wet road) |

Dash odometry covers the two highway clips (`changelane` 83%, `too_close` 86%).
Known limits: at highway speed the road moves too far between frames to match
reliably, and on wet roads reflections barely move and win the match.

The report's own accuracy table is stricter: it counts unmatched frames as
0 km/h and pads the truth with zeros after its last reading, so road odometry,
which leaves more frames unmeasured, reads lower there than in the table above.

To see what dash odometry is detecting, open `dash_detections.jpg`; for a clip
with ground truth, `dash_odometry_vs_ground_truth.png` plots speed, distance and
path against the dashcam's readout.

## 4. Frames to video

Set `frame_dir`, `output_name`, `fps` at the top of `frames_to_video.py`:

```bash
python $BASE/scripts/vis3d/frames_to_video.py
```

Writes an H.264 mp4 into `scripts/vis3d/videos/`.

## 5. Export the dataset to navsim logs

`generate_ras_logs.py` collects the rendered runs, scattered across
`data/<dataset>/<clip>/<run>/`, into the single navsim-format tree a training run
reads:

```bash
conda activate rap                                 # NOT vis3d -- see setup step 6
python $BASE/scripts/vis3d/generate_ras_logs.py --dry-run   # read everything, write nothing
python $BASE/scripts/vis3d/generate_ras_logs.py
```

There is no CLI beyond `--dry-run`; edit the CONFIG block at the top instead.
Output goes to `$BASE/CARE` by default, laid out like
`carla_garage_data_navsim_converted` so these and the CARLA logs load through the
same `SceneLoader`:

    CARE/
    ├── openscene_meta_datas/<scene_token>.pkl       one dict per frame
    ├── sensor_blobs/<log>/CAM_F0/<token>.jpg        the photograph
    ├── rendered_sensor_blobs/<log>/CAM_F0/<token>.jpg   the rasterization
    └── synthetic_scene_pickles/<scene_token>.pkl    two-stage eval stub

Point a training run at it with `sim_log_path: <root>/openscene_meta_datas` and
`sim_sensor_path: <root>/sensor_blobs`. The rendered tree is never configured:
`navsim/common/dataclasses.py:77` derives it from the sensor path by substituting
`sensor_blobs` -> `rendered_sensor_blobs`, so the two trees must agree filename
for filename and neither can be renamed.

### What to set

`CLIP_RUNS` is the list that decides what gets exported, as `(dataset, clip, run)`
triples:

```python
CLIP_RUNS = [
    ("CARE_YTB", "*", "1"),
]
```

A clip of `"*"` means every clip in that dataset rendered at that run; clips that
have not been are listed and skipped, not exported as empty logs.

The `run` has to name the run the pipeline actually delivered -- with the 10 Hz
default that is **`2hz`**. `"1"` works only through the `1 -> 2hz` symlink
`run_render_vis3d.sh` makes when that name is free, so prefer `"2hz"`.

Four more entries are worth knowing before the first export:

- **`REQUIRE_EGO_POSES = True`** skips a clip with no `ego_poses.txt`. The pose
  track *is* the label -- `Scene.get_future_trajectory` is built from nothing
  else -- so leave it on for anything that will be trained on.
- **`CORRECT_ROAD_PLANE = True`** re-levels the ego frame by fitting the road
  plane from the boxes sitting on it, correcting the lift's assumption that the
  optical axis is horizontal (measured at +6.3 deg on `blocker`, a metre of false
  box height every ten metres). Rigid, so reprojection is unaffected.
- **`RENDERED_PAD_ROWS = 20`** adds black rows top and bottom of every rendered
  frame, because navsim reads rendered views as `Image.open(...)[20:-20]`
  unconditionally. Without it every row sits 20 px from where the real image puts
  it. Set to 0 only if the renderer starts emitting the taller canvas itself.
- **`PHASES = (0,)`** picks which interleavings of the 2 Hz subsample to keep.
  Each is a valid sequence of the same clip under its own log name, so listing
  more multiplies the scene count with content that overlaps.

`LINK_SENSORS = True` symlinks the images rather than copying them; set it False
if the export has to survive its clip directories being moved or deleted.

### Two constraints on the clips

The export prints both on exit, but they are decisions made when picking clips:

- **A scene needs `num_history_frames + num_future_frames` frames** -- 14 for
  navtrain, which at 2 Hz is **7 seconds of footage**. A shorter clip exports
  cleanly and yields no scenes at all.
- **Only `CAM_F0` is written.** `rap_agent.py` asks for `cam_f0/l0/r0/b0`; the
  three a dashcam does not have load as zeros. Expected, not a gap to fill.

### Why the dry run matters here

Every way of getting this wrong is silent: `dataclasses.py:79-82` swallows an
unreadable rendered image into `np.zeros((1080,1920,3))` behind a bare `except`,
and a missing real image becomes `zeros_like` with `real_valid` False. A tree
that is misnamed or short a frame trains on black images and reports nothing --
not at export, not at load, not during training. `--dry-run` reads the same
inputs and fails on the same missing files as a real export.

## Example data layout

`data/YTB/beepbeep/`

```
beepbeep/
├── beepbeep.mp4                     # as downloaded
├── frames/                          # 163 × 000000.jpg …
├── mask_results_preds.json          # clip-level run (RUN="")
├── boxes_3d.json
├── 1/                               # RUN="1"
│   ├── mask_results_preds.json      # stage 1
│   ├── vis/                         # stage 1, --save_vis: 163 annotated jpgs
│   ├── lane_masks/                  # stage 1b: 163 × 000000.png
│   ├── samples-pseudodepth/         # stage 2: 163 × {000000.xyz.npy, 000000.K.npy}
│   ├── boxes_3d.json                # stage 3
│   ├── vis3d/                       # stage 3: 163 rasterized jpgs
│   ├── vis3d_overlay/               # stage 3: 163 raster-over-frame jpgs
│   ├── ego_poses.txt                # stage 4
│   ├── ego_trajectory.png           # stage 4: the sanity plot
│   ├── camera_intrinsics.json       # run_odometry.sh: intrinsics
│   ├── road_plane.json              # run_odometry.sh: plane (+ quality)
│   ├── dash_speed.json              # run_odometry.sh: dash odometry
│   ├── road_speed.json              # run_odometry.sh: road odometry
│   ├── ego_speed.json               # run_odometry.sh: the chosen speed
│   ├── navsim_logs/                 # export: <split>/<clip>.pkl
│   └── sensor_blobs/                # export: CAM_F0 symlinks into ../frames/
└── 2/                               # RUN="2", reusing run 1
    ├── mask_results_preds.json -> ../1/mask_results_preds.json
    ├── lane_masks -> ../1/lane_masks
    ├── samples-pseudodepth -> ../1/samples-pseudodepth
    ├── boxes_3d.json
    ├── vis3d/
    └── vis3d_overlay/
```

That is the pre-10 Hz layout, kept because `beepbeep` is still in it. A clip
rendered by the current pipeline also carries `frames_10hz/` + `10hz/`, with
`frames/` and `1/` linked to the delivered `frames_2hz/` + `2hz/`; the per-run
contents are the same either way.

`navsim_logs/` and `sensor_blobs/` *inside a run* are stage 4's per-clip export,
not what training reads -- step 5 builds that.

## Deprecated: OpenVO

OpenVO, the learned visual-odometry model vendored under `vis3d/openvo/`, is no
longer used by the pipeline. `run_render_dataset.sh` runs stage 4 with
`VO_METHOD=pointcloud`, and metric ego speed now comes from `run_odometry.sh`
(section 3c).

Why: it takes stage 2's depth as input, and on 18 of the 50 CARE clips that
depth collapses -- the ego's own bonnet comes back 25-30 m away. OpenVO then
regresses almost no motion (`changelane`: about 10 m of path where the dashcam's
GPS says 265 m), and its per-frame speed has essentially no correlation with the
burned-in speed readout (r = 0.07). Swapping the depth model for Metric3D failed
on the same clips.

The code is still there. `run_render_vis3d.sh`'s own default is still
`VO_METHOD=openvo` when run directly, and `estimate_ego_motion.py --method openvo`
still works if the builds below are present.

### Setup (only if you still need it)

Two of its dependencies are source builds:

```bash
conda activate vis3d

# pytorch3d -- CPU-only is enough; only pytorch3d.transforms is used, and that
# is pure torch. FORCE_CUDA=0 keeps the build to a few minutes.
FORCE_CUDA=0 pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8"

# the correlation CUDA extension, rebuilt against this env's torch
cd $BASE/vis3d/openvo/model/correlation_package
git apply $BASE/scripts/vis3d/openvo-correlation-cxx17.patch   # c++14 -> c++17
CUDA_HOME=/opt/common/cuda/cuda-12.1.1 PATH=$CUDA_HOME/bin:$PATH \
  pip install --no-build-isolation .
```

The extension used to ship as a `cpython-39` `.so`, which is what kept stage 4
in its own py3.9 / torch-2.0.1 env. The CUDA source needed no changes at all --
only the `-std=c++14` flag in `setup.py`, which torch >= 2.1 headers reject
outright. Use a `CUDA_HOME` matching torch's build (cu121 here); building needs
nvcc but not a GPU.

Verify:

```bash
python -c "import pytorch3d.transforms, correlation_cuda; print('openvo ok')"
```

### Running it

Stage 4 of `run_render_vis3d.sh` with the learned model, on a clip whose point
maps already exist:

```bash
RUN_STAGE1_MASKS=0 RUN_STAGE1B_LANES=0 RUN_STAGE2_DEPTH=0 RUN_STAGE3_LIFT=0 \
RUN_STAGE4_VO=1 VO_METHOD=openvo EXPORT_NAVSIM=0 \
VIDEO=<clip> DATASET=<dataset> RUN=1 \
sbatch --export=ALL $BASE/scripts/vis3d/run_render_vis3d.sh
```

It needs a GPU and writes `ego_poses.txt` and `ego_trajectory.png` into the run,
overwriting the point-cloud ones.
