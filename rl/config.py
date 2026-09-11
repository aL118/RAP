"""Configuration for iterative scorer training. One dataclass, all defaults runnable.

The pipeline this configures is not PPO. It is iterative distillation of the true
PDM score into RAP's own `Scorer` head (rounds of collect -> score -> train), so
the knobs are about *rounds and buffers*, not about policy gradients.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple
import os


def _devkit_root() -> Path:
    """RAP checkout root. The sbatch scripts export NAVSIM_DEVKIT_ROOT; fall back
    to this file's grandparent so interactive use needs no environment at all."""
    return Path(os.getenv("NAVSIM_DEVKIT_ROOT", Path(__file__).resolve().parents[1]))


@dataclass
class RLConfig:
    """Everything the iterative pipeline needs. Field groups map 1:1 onto modules."""

    # =================================================================== data
    # ---- clips: the CARE export, which is what the rounds are actually about.
    # generate_ras_logs.py writes these three trees side by side. They are plain
    # list-of-dict pickles in openscene format, one per clip, and rl/care.py reads
    # them directly rather than through navsim's SceneLoader -- see the module
    # docstring there for why (Scene._build_map_api asserts on the clip's map name).
    care_root: Path = field(default_factory=lambda: _devkit_root() / "CARE")

    # Frames per scene window and the stride between windows. A CARE clip is ~74
    # frames at 0.5 s, so stride 1 turns one clip into ~61 overlapping scenes and
    # stride 14 into 5 disjoint ones. 1 is the default because 50 clips is a small
    # pool and every window is a different situation to score; raise it if the
    # correlation between neighbouring windows starts to matter.
    num_history_frames: int = 4
    num_future_frames: int = 10
    care_frame_stride: int = 1

    # Which camera slots a CARE window fills, indexing `ImgEncoder.cams_embeds`.
    # The order that table is indexed by is the iteration order in
    # `LoadMultiViewImageFromFiles`: (cam_b0, cam_f0, cam_l0, cam_r0). CARE writes
    # CAM_F0 only, so the one image belongs at index 1 -- NOT 0, which is the rear
    # camera and is what the unpatched encoder would silently give it. See
    # rl/model.py:patch_image_encoder_camera_embeddings.
    care_camera_indices: Tuple[int, ...] = (1,)

    # ---- regular data: navtrain, mixed in so the rounds do not only ever see
    # crash footage. Without it the scorer is trained exclusively on clips where
    # the *human* collides, and the cheapest way to score well on those is to
    # prefer trajectories that barely move. The navtrain mix carries ordinary
    # driving where progress is rewarded, which is what keeps the planner honest.
    navtrain_cache_path: Path = field(
        default_factory=lambda: _devkit_root() / "cache" / "rap_ego"
    )
    # PDM metric cache for the navtrain half. Agent-independent: it stores the
    # simulation/scoring state of the scenario and any trajectory is scored against
    # it afterwards. Reused from DrivoR rather than rebuilt.
    metric_cache_path: Path = field(
        default_factory=lambda: Path(
            os.getenv("RAP_METRIC_CACHE", _devkit_root() / "exp" / "train_metric_cache")
        )
    )
    # navtrain scenes drawn per round, as a multiple of the clip count. 1.0 means
    # "as many navtrain scenes as CARE scenes". They are redrawn each round from a
    # seeded RNG, so across 4 rounds the buffer sees 4x this many distinct scenes.
    regular_ratio: float = 1.0

    # ---- the pretrained planner every round 1 sample comes from.
    checkpoint_path: Path = field(
        default_factory=lambda: _devkit_root() / "weights" / "RAP_DINO_navsimv2.ckpt"
    )
    # Must match the checkpoint. RAP_DINO_navsimv2.ckpt was trained at 5 s / 0.5 s,
    # i.e. 10 poses. The PDM scorer consumes the first 8 of them (4 s).
    time_horizon: float = 5.0
    interval_length: float = 0.5

    # Fraction of scenes held out from every round. Deterministic (hash of token),
    # so collect.py, train.py and eval.py agree without writing a split file.
    val_fraction: float = 0.1

    # ================================================================= rounds
    # Round 1 samples from the pretrained model; rounds 2..N from the model the
    # previous round produced. 4 sits in the middle of the 3-5 the design calls for.
    num_rounds: int = 4

    # Trajectories sampled per scene per round, on top of the human trajectory.
    # RAP emits proposal_num (64) proposals in one forward pass, so this is a
    # subsample of those rather than repeated sampling -- see rl/collect.py.
    samples_per_scene: int = 32
    # Of those, how many are the model's own top-ranked proposals (the ones its
    # current scorer would actually pick, i.e. the on-policy part) versus drawn
    # uniformly from the rest (the coverage part). Splitting the two is the whole
    # point: train only on the top-k and the scorer never learns why the proposals
    # it rejects are bad, which is exactly the distribution it has to rank at
    # inference.
    top_k_fraction: float = 0.5

    # The human trajectory is added once, in round 1, and never resampled: it does
    # not depend on the model, so re-adding it every round would just reweight the
    # buffer towards it by a factor of num_rounds.
    include_human: bool = True

    # ================================================================ scoring
    # Sub-scores, in the order compute_navsim_score.get_sub_score stacks them.
    # Both backends emit this layout; the CARE one marks columns it cannot label.
    #
    # CARE clips carry no map -- map_location is the clip's own name and
    # roadblock_ids is empty -- so drivable_area_compliance has no ground truth
    # there. The lane polylines in the export are not a substitute: they are
    # per-frame detections with no temporal association and a known over-extension
    # bug, so a drivable label built from them would be confidently wrong rather
    # than missing. Left unlabelled, the BCE loss masks that column for CARE rows
    # and the navtrain half of the buffer supplies it.
    care_label_drivable: bool = False

    # Processes used to PDM-score a round's trajectories. The scorer is a stateful
    # module-level singleton (see rl/scoring.py), so this is a process count and
    # never a thread count.
    score_workers: int = 12

    # ============================================================== training
    # Where round r starts from. False continues from round r-1's weights, which is
    # cheaper and is what DAgger normally does. True re-fits the pretrained
    # checkpoint on the whole buffer every round instead: slower, but the rounds
    # stop compounding -- round 4 is then "the pretrained planner fitted to four
    # rounds of data" rather than "four fine-tunes stacked on each other", and a
    # bad round cannot poison the ones after it.
    restart_from_pretrained: bool = False

    # Write each round's checkpoint as a delta against the pretrained one rather
    # than a self-contained 3.5 GB file. With freeze_backbone the DINOv3 tower is
    # 95% of that and is identical in every round, so this is ~190 MB per round.
    # rl/export.py turns one back into a full checkpoint when something outside
    # this pipeline needs to load it.
    save_full_checkpoints: bool = False

    epochs_per_round: int = 4
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    num_workers: int = 4

    # Loss weights. sub_score covers all six columns, final_score singles out the
    # aggregate PDMS column that inference actually ranks on.
    sub_score_weight: float = 1.0
    final_score_weight: float = 1.0
    # Imitation loss on the regular-data half only. The clips' human trajectories
    # are the *crash* trajectories, so training the planner to reproduce them is
    # the one thing this pipeline must not do; rl/train.py applies this weight to
    # navtrain rows and zero to CARE rows.
    trajectory_weight: float = 1.0

    # The ViT is frozen in RAP training anyway. Freezing it here as well keeps the
    # rounds cheap and means the scene features do not drift between the round that
    # scored a trajectory and the round that trains on it.
    freeze_backbone: bool = True

    # Candidate trajectories are scored by re-running the last refiner stage with
    # the candidates as its reference points (rl/model.py). That stage has exactly
    # proposal_num query slots, so a round with fewer candidates pads to it and
    # masks the padding out of the loss. More candidates than slots is a hard error
    # rather than a silent truncation.
    max_candidates: Optional[int] = None  # None -> the model's proposal_num

    # ================================================================ runtime
    seed: int = 0
    experiment_name: str = "rap_iterative_scorer"

    # Per-batch scalars to `<run>/logs/*.jsonl` and TensorBoard event files under
    # `<run>/tb/`. The JSONL is written either way and is the durable record; this
    # only turns off the event files, which cost a `torch.utils.tensorboard`
    # import (a few seconds, it pulls TF in this env) and nothing per step.
    log_tensorboard: bool = True

    @property
    def num_poses(self) -> int:
        """Poses in a trajectory. Derived, not configured: it has to agree with the
        checkpoint's init_feature embedding or nothing loads."""
        return int(round(self.time_horizon / self.interval_length))

    @property
    def output_path(self) -> Path:
        """Run directory: per-round checkpoints, logs, eval output."""
        exp_root = Path(os.getenv("NAVSIM_EXP_ROOT", _devkit_root() / "exp"))
        return exp_root / self.experiment_name

    @property
    def buffer_path(self) -> Path:
        """The append-only buffer. Lives inside the run directory because it *is*
        the run: a round trains on every round before it, so a buffer from another
        experiment is not interchangeable."""
        return self.output_path / "buffer"

    def round_checkpoint(self, round_index: int) -> Path:
        """Weights produced by round `round_index` (1-based). Round 0 is the
        pretrained checkpoint the whole thing starts from."""
        if round_index <= 0:
            return self.checkpoint_path
        return self.output_path / f"round_{round_index:02d}" / "model.pt"

    def validate(self, require_buffer: bool = False) -> None:
        """Fail early with an actionable message instead of deep inside a worker."""
        missing = []
        if not self.checkpoint_path.is_file():
            missing.append(f"{self.checkpoint_path} -- pretrained RAP weights")
        if not (self.care_root / "openscene_meta_datas").is_dir():
            missing.append(
                f"{self.care_root / 'openscene_meta_datas'} -- run "
                f"scripts/vis3d/generate_ras_logs.py"
            )
        if self.regular_ratio > 0:
            if not self.navtrain_cache_path.is_dir():
                missing.append(f"{self.navtrain_cache_path} -- run_dataset_caching.sh")
            if not self.metric_cache_path.is_dir():
                missing.append(f"{self.metric_cache_path} -- set RAP_METRIC_CACHE")
        if require_buffer and not self.buffer_path.is_dir():
            missing.append(f"{self.buffer_path} -- run rl/collect.py first")
        if missing:
            raise FileNotFoundError("Missing required input(s):\n  " + "\n  ".join(missing))

    def sub_score_names(self) -> Tuple[str, ...]:
        from rl.scoring import SCORE_KEYS

        return SCORE_KEYS
