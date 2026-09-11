"""The RAP planner, wrapped so that arbitrary trajectories can be scored.

The problem this module solves
------------------------------
`Scorer` does not take a trajectory. It takes the refiner stack's query features,
pools them per proposal and reads six sub-scores off an MLP:

    proposal_feature = bev_feature.reshape(B, P, T, -1).amax(-2)
    pred_logit       = self.pred_score(proposal_feature)

The trajectory only enters through `Bev_refiner`, which uses the *poses as
reference points* for its deformable attention: `ref_2d = pose.detach()`, later
normalised as `(ref_2d[..., :2] + 32) / 64` over the +-32 m point-cloud range, with
the heading driving `compute_corners`. So a proposal's feature is "the query slot,
having attended to the image at the places this trajectory goes".

That is the hook. To score a trajectory the model did not propose -- the human's,
or one a previous round sampled and the buffer kept -- run the final refiner stage
with that trajectory as its reference points instead of the model's own proposal:

    F_in            = features entering the last Traj_refiner stage
    F_candidate     = Bev_refiner(candidate, F_in, image_feature)
    logits          = pred_score(pool(F_candidate))

When `candidate` is the model's own final proposal this reproduces
`RAPModel.forward` exactly, which is the property that makes it the right hook:
the scorer is trained through the same path it is used through at inference, so
nothing about the buffer's off-policy rows is scored by a different mechanism than
the ranking they are supposed to improve.

Why the features cannot be cached instead
-----------------------------------------
The refiners train. A trajectory scored in round 1 has to be re-featurised by the
round-3 model before round 3 can learn from it, so the buffer stores trajectories
and their true scores, never features. That is also why the ViT stays frozen: the
image tower is the one part that could be cached, and freezing it means the round
that scored a trajectory and the round that trains on it at least agree about what
the scene looks like.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# Camera order in `LoadMultiViewImageFromFiles`, which is what indexes
# `ImgEncoder.cams_embeds`. A CARE window fills only CAM_F0, i.e. index 1.
CAMERA_ORDER = ("cam_b0", "cam_f0", "cam_l0", "cam_r0")


def patch_image_encoder_camera_embeddings() -> None:
    """Stop a single-camera input from being broadcast into four.

    `ImgEncoder.forward` adds the learned per-camera embedding with

        feat = feat + self.cams_embeds[:, None, None, :]

    where `feat` is (num_cam, bs, h*w, C) and `cams_embeds` is (4, C). With four
    cameras that is the intended elementwise add. With **one** -- which is every
    CARE window, because `generate_ras_logs.py` writes CAM_F0 and nothing else --
    numpy-style broadcasting silently turns (1, bs, hw, C) into (4, bs, hw, C):
    the one real image is replicated four times, each copy tagged with a different
    camera's embedding.

    Nothing raises, and the run does not obviously break, which is what makes it
    worth patching rather than leaving. Two things are wrong downstream:

      * `point_sampling` derives its reference points from `lidar2img`, which has
        one camera, so only index 0 of the replicated stack is ever attended to --
        and index 0 is CAM_B0. The CARE front image is fed through the model
        wearing the *back* camera's embedding. The geometry is right (lidar2img is
        the front camera's); the learned offset is the wrong one.
      * the other three copies cost a full deformable-attention pass each and
        contribute nothing.

    So this replaces the tail of `ImgEncoder.forward` with the same computation,
    indexing `cams_embeds` by the cameras actually present. The indices come from
    `_rl_camera_indices` on the module, which `RAPScorer` sets from
    `config.care_camera_indices`. Advanced indexing keeps the gradient, so the
    embedding still trains.

    A four-camera batch takes the original method unchanged, so the navtrain path
    is bit-identical to the supervised pipeline's.

    NOTE: this and `patch_spatial_cross_attention_num_cams` are a pair. The
    broadcast above is what currently makes the hardcoded `num_cams=4` in
    `SpatialCrossAttention` line up by accident; removing it leaves a genuinely
    one-camera feature, which that reshape cannot handle. Applying either alone is
    worse than applying neither.
    """
    from navsim.agents.rap_dino.bevformer.image_encoder import ImgEncoder

    if getattr(ImgEncoder.forward, "_rl_camera_patched", False):
        return

    original = ImgEncoder.forward

    def forward(self, img, len_queue=None, **kwargs):
        if img is None or img.size(1) == self.cams_embeds.shape[0]:
            return original(self, img, len_queue, **kwargs)
        if len_queue is not None:
            raise NotImplementedError(
                "len_queue is a temporal-BEV path this pipeline never takes"
            )

        batch, num_cameras, channels, height, width = img.size()
        flat = img.reshape(batch * num_cameras, channels, height, width)
        if self.training and self.use_grid_mask:
            flat = self.grid_mask(flat)

        features = self._tokens_to_map(
            self._backbone_forward(flat), batch, num_cameras, height, width
        )
        if isinstance(features, dict):
            features = list(features.values())
        if self.with_img_neck:
            features = self.img_neck([features])

        # num_outs is 1, so there is exactly one level and lvl is 0 throughout.
        feature = features[-1]
        fused, embed_dims, grid_h, grid_w = feature.size()
        feature = feature.view(batch, fused // batch, embed_dims, grid_h, grid_w)
        feature = feature.flatten(3).permute(1, 0, 3, 2)  # (num_cam, bs, h*w, C)

        if self.use_cams_embeds:
            indices = getattr(self, "_rl_camera_indices", None)
            if indices is None:
                indices = list(range(feature.shape[0]))
            if len(indices) != feature.shape[0]:
                raise ValueError(
                    f"{feature.shape[0]} cameras in the batch but "
                    f"_rl_camera_indices names {len(indices)}: {indices}"
                )
            feature = feature + self.cams_embeds[list(indices)][:, None, None, :].to(
                feature.dtype
            )
        feature = feature + self.level_embeds[None, None, 0:1, :].to(feature.dtype)

        spatial_shape = torch.as_tensor(
            [(grid_h, grid_w)], dtype=torch.long, device=feature.device
        )
        level_start_index = torch.cat(
            (spatial_shape.new_zeros((1,)), spatial_shape.prod(1).cumsum(0)[:-1])
        )
        return feature.permute(0, 2, 1, 3), spatial_shape, level_start_index, kwargs

    forward._rl_camera_patched = True
    ImgEncoder.forward = forward


def patch_spatial_cross_attention_num_cams() -> None:
    """Let `SpatialCrossAttention` run with a camera count other than four.

    `Bev_refiner` hardcodes `num_cams=4` in its attention config, and the forward
    pass reshapes with that constant:

        key.permute(2, 0, 1, 3).reshape(bs * self.num_cams, l, self.embed_dims)

    Every other quantity in that function -- `bev_mask`, `reference_points_cam`,
    the `indexes` list -- is sized from the number of cameras `point_sampling`
    found in `lidar2img`. On navtrain that is 4 and the constant is right. Once
    `patch_image_encoder_camera_embeddings` stops replicating a single camera into
    four, a CARE window genuinely has one, and this reshape asks for four times the
    elements it has.

    Nothing about the weights depends on the camera count: the deformable attention
    is parameterised per head, level and sampling point, and the cameras are only a
    batch dimension. So the fix is to use the count that is actually present. It
    belongs in `spatial_cross_attention.py` as `num_cams = key.shape[0]`, but this
    package does not edit `navsim/`, so it is applied here as a wrapper that sets
    the attribute for the duration of the call and restores it afterwards.

    Idempotent, and safe to call before every run: a second call is a no-op.
    """
    from navsim.agents.rap_dino.bevformer.spatial_cross_attention import (
        SpatialCrossAttention,
    )

    if getattr(SpatialCrossAttention.forward, "_rl_num_cams_patched", False):
        return

    original = SpatialCrossAttention.forward

    def forward(self, *args, **kwargs):
        key = kwargs.get("key", args[1] if len(args) > 1 else None)
        if key is None:
            return original(self, *args, **kwargs)
        previous, self.num_cams = self.num_cams, int(key.shape[0])
        try:
            return original(self, *args, **kwargs)
        finally:
            self.num_cams = previous

    forward._rl_num_cams_patched = True
    SpatialCrossAttention.forward = forward


def build_rap_config(config):
    """RAPConfig at the horizon the checkpoint was trained with."""
    from navsim.agents.rap_dino.navsim_config import RAPConfig
    from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

    rap_config = RAPConfig()
    rap_config.trajectory_sampling = TrajectorySampling(
        time_horizon=config.time_horizon, interval_length=config.interval_length
    )
    return rap_config


def load_state_dict(model: nn.Module, checkpoint_path: Path) -> None:
    """Load a full checkpoint into `RAPModel`.

    The pretrained weights come from a Lightning run and are prefixed
    `agent._rap_model.*`; a full round checkpoint is already bare. Both are
    accepted so a resume and a cold start take the same path. Delta round
    checkpoints go through `RAPScorer.from_checkpoint` instead, which loads their
    base first.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {
        key.replace("agent._rap_model.", "").replace("_rap_model.", ""): value
        for key, value in state_dict.items()
    }

    # strict=False is needed for the training-only domain_classifier, but it also
    # means a horizon mismatch would pass silently: init_feature is
    # (num_poses * proposal_num, d_model), so loading a 10-pose checkpoint into an
    # 8-pose model would skip that key and leave a randomly initialised embedding
    # driving every trajectory. Check it explicitly.
    expected = model.init_feature.weight.shape
    found = state_dict.get("init_feature.weight")
    if found is not None and tuple(found.shape) != tuple(expected):
        raise ValueError(
            f"Checkpoint/config mismatch: init_feature is {tuple(found.shape)} in "
            f"{Path(checkpoint_path).name} but {tuple(expected)} in this config. "
            f"Pass the --time-horizon the checkpoint was trained with "
            f"(RAP_DINO_navsimv2.ckpt used 5)."
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[model] {len(missing)} missing keys, first few: {missing[:5]}")
    if unexpected:
        print(f"[model] {len(unexpected)} unexpected keys, first few: {unexpected[:5]}")


class RAPScorer(nn.Module):
    """RAP with two entry points: propose trajectories, and score given ones.

    `RAPAgent` is bypassed on purpose. Its constructor builds metric-cache loaders,
    loss modules and optionally a ray pool, none of which this pipeline uses -- the
    scoring happens in a separate process pool over the buffer, not inline.
    """

    def __init__(self, config, device: Optional[torch.device] = None):
        super().__init__()
        from navsim.agents.rap_dino.rap_model import RAPModel

        # Applied as a pair; see the docstrings above for why neither is safe alone.
        patch_image_encoder_camera_embeddings()
        patch_spatial_cross_attention_num_cams()

        self.config = config
        self.rap_config = build_rap_config(config)
        self.rap = RAPModel(self.rap_config)
        self.num_proposals = self.rap_config.proposal_num
        self.num_poses = self.rap.poses_num
        self.device = device or torch.device("cpu")

        # Which slots of `cams_embeds` a single-camera batch should use. Only read
        # when the batch has fewer cameras than the embedding table has rows, i.e.
        # on CARE windows; a navtrain batch never reaches it.
        self.rap._backbone._rl_camera_indices = list(config.care_camera_indices)

    # ------------------------------------------------------------------ building

    @classmethod
    def from_checkpoint(cls, config, checkpoint_path: Path, device: torch.device):
        """Build and load, transparently handling delta round checkpoints.

        A round checkpoint written with `save_full=False` holds only the tensors
        that round could actually change -- everything frozen is excluded, which on
        the default config is the 840M-parameter DINOv3 tower, i.e. 95% of the
        file. It records the base it was trained from and is applied on top of it
        here. `load_state_dict` alone would not do: it uses `strict=False` (needed
        for the training-only `domain_classifier`), so handing it a delta would
        silently leave a randomly initialised ViT driving every trajectory.
        """
        model = cls(config, device)
        checkpoint_path = Path(checkpoint_path)

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(checkpoint, dict) and checkpoint.get("rl_partial"):
            base = Path(checkpoint["base_checkpoint"])
            if not base.is_file():
                raise FileNotFoundError(
                    f"{checkpoint_path} is a delta checkpoint whose base "
                    f"{base} is missing. It holds only the tensors its round "
                    f"trained; the rest has to come from the base. Point "
                    f"config.checkpoint_path at the original and re-export, or "
                    f"re-run that round with --save-full."
                )
            load_state_dict(model.rap, base)
            missing, unexpected = model.rap.load_state_dict(
                checkpoint["state_dict"], strict=False
            )
            if unexpected:
                raise ValueError(
                    f"{checkpoint_path} carries {len(unexpected)} tensors this "
                    f"model has no slot for, first few: {unexpected[:5]}"
                )
            print(f"[model] {checkpoint_path.name}: applied "
                  f"{len(checkpoint['state_dict'])} tensors over {base.name}")
        else:
            load_state_dict(model.rap, checkpoint_path)

        model.to(device)
        if config.freeze_backbone:
            for parameter in model.rap._backbone.img_backbone.parameters():
                parameter.requires_grad = False
        return model

    def trainable_parameters(self):
        """Everything the rounds update: the refiner stack and the scorer head.

        The ViT is excluded by `freeze_backbone`; the domain classifier is excluded
        because it belongs to the supervised pipeline's adversarial term and has no
        gradient here.
        """
        excluded = ("_backbone.img_backbone", "domain_classifier")
        return [
            parameter
            for name, parameter in self.rap.named_parameters()
            if parameter.requires_grad and not any(part in name for part in excluded)
        ]

    def save(self, path: Path, full: bool = False) -> None:
        """Write this round's weights.

        By default only the tensors that were trainable are written, with the base
        checkpoint recorded alongside. With `freeze_backbone` that is ~190 MB
        instead of 3.5 GB, and four rounds cost 760 MB rather than 14 GB of a
        shared filesystem. The frozen tower is then byte-identical across rounds by
        construction rather than by four copies happening to agree.

        `full=True` writes a self-contained file. Use it when the checkpoint has to
        leave this pipeline -- `rl/export.py` is the other way to get one.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        state_dict = self.rap.state_dict()

        if not full:
            frozen = {
                name for name, parameter in self.rap.named_parameters()
                if not parameter.requires_grad
            }
            state_dict = {
                key: value for key, value in state_dict.items() if key not in frozen
            }

        payload = {"state_dict": state_dict}
        if not full:
            payload["rl_partial"] = True
            payload["base_checkpoint"] = str(Path(self.config.checkpoint_path).resolve())
        torch.save(payload, path)

    # ------------------------------------------------------------------- forward

    def encode(self, features: Dict[str, torch.Tensor]):
        """Run the refiner stack up to -- but not through -- its final refinement.

        This is `RAPModel.forward` unrolled, stopping one step early. The last
        `Traj_refiner` call does two things: decode proposals from the incoming
        query features, then refine those features at the decoded poses. Only the
        decode is unconditional; which poses the refinement uses is exactly the
        choice this module exists to make, so it is left to the caller
        (`propose` uses the model's own proposals, `score_candidates` uses the
        buffer's trajectories) and neither pays for the other's refinement.

        It is not a reimplementation of the maths -- every line calls the model's
        own submodules -- but it is a second copy of the call order, so a change to
        `RAPModel.forward` or `Traj_refiner.forward` has to be mirrored here.

        :return: ``(image_feature, feature_in, proposal_list)`` -- the query
            features entering the last stage, and every stage's proposals, the
            last entry being the ones decoded from `feature_in`. The whole list is
            returned because RAP's own trajectory loss supervises every stage, and
            the decodes are one MLP each.
        """
        ego_status = features["ego_status"][:, -1]
        if self.rap.b2d:
            ego_status[:, 1:3] = 0

        image_feature = self.rap._backbone(features["camera_feature"], img_metas=features)

        bev_feature = self.rap.hist_encoding(ego_status)[:, None] + self.rap.init_feature.weight[None]

        proposal_list = []
        stages = self.rap._trajectory_head
        for refiner in stages[:-1]:
            bev_feature, proposal_list = refiner(bev_feature, proposal_list, image_feature)

        # The decode half of stages[-1].forward, verbatim.
        last = stages[-1]
        proposal_list.append(
            last.traj_decoder(bev_feature).reshape(
                bev_feature.shape[0], -1, last.poses_num, last.state_size
            )
        )
        return image_feature, bev_feature, proposal_list

    def _score_head(self, bev_feature: torch.Tensor) -> torch.Tensor:
        """Sub-score logits from refined query features.

        Mirrors the first three statements of `Scorer.forward`. The rest of that
        method -- the agent, area and BEV-semantic heads -- runs only under
        `self.training` and needs targets this pipeline does not produce (they come
        from the metric cache's key-agent bookkeeping, which CARE clips have no
        equivalent of), so it is skipped rather than computed and discarded.
        """
        batch_size = bev_feature.shape[0]
        proposal_feature = bev_feature.reshape(
            batch_size, self.num_proposals, self.num_poses, -1
        ).amax(-2)

        logit = self.rap.scorer.pred_score(proposal_feature).reshape(
            batch_size, -1, self.rap.scorer.score_num
        )
        if self.rap.scorer.double_score:
            logit2 = self.rap.scorer.pred_score2(proposal_feature).reshape(
                batch_size, -1, self.rap.scorer.score_num
            )
            # The model averages the two heads' probabilities, not their logits, so
            # this averages in probability space too and converts back.
            probability = (torch.sigmoid(logit) + torch.sigmoid(logit2)) / 2
            logit = torch.logit(probability.clamp(1e-6, 1 - 1e-6))
        return logit

    def propose(self, features: Dict[str, torch.Tensor]):
        """The model's own proposals and how it currently ranks them.

        :return: ``(proposals (B, N, P, 3), pdm_score (B, N))`` -- the same pair
            `RAPModel.forward(..., return_score=True)` returns, so the trajectory a
            round samples is drawn from exactly the distribution inference sees.
        """
        image_feature, feature_in, proposal_list = self.encode(features)
        proposals = proposal_list[-1]
        refined = self.rap._trajectory_head[-1].Bev_refiner(
            proposals, feature_in, image_feature
        )
        pdm_score = torch.sigmoid(self._score_head(refined))[..., -1]
        return proposals, pdm_score

    def forward_train(
        self, features: Dict[str, torch.Tensor], candidates: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """Proposals and candidate logits from a single encode.

        A training step needs both: the candidate logits carry the score loss, and
        the model's own proposals carry the imitation loss that keeps the planner
        from drifting while the scorer is being retrained. Running `propose` and
        `score_candidates` separately would pay for the ViT and three refiner
        stages twice, which on this model is most of the step.

        :return: ``(proposal_list, logits (B, K, 6))`` -- every refinement stage's
            proposals, for the staged trajectory loss, and the candidate logits.
        """
        num_candidates = candidates.shape[1]
        self._check_candidates(candidates)

        image_feature, feature_in, proposal_list = self.encode(features)
        refined = self.rap._trajectory_head[-1].Bev_refiner(
            self._pad_candidates(candidates), feature_in, image_feature
        )
        return proposal_list, self._score_head(refined)[:, :num_candidates]

    def _check_candidates(self, candidates: torch.Tensor) -> None:
        if candidates.shape[1] > self.num_proposals:
            raise ValueError(
                f"{candidates.shape[1]} candidates but the refiner has "
                f"{self.num_proposals} query slots. Lower samples_per_scene or "
                f"max_candidates, or score in chunks."
            )
        if candidates.shape[2] != self.num_poses:
            raise ValueError(
                f"candidates have {candidates.shape[2]} poses, the model wants "
                f"{self.num_poses} -- check config.time_horizon against the checkpoint."
            )

    def _pad_candidates(self, candidates: torch.Tensor) -> torch.Tensor:
        """Fill the refiner's fixed query grid by tiling the real candidates.

        `Bev_refiner`'s positional encoding is built over a fixed
        `proposal_num x num_poses` grid and the query tensor has exactly that many
        rows, so a batch with fewer candidates still has to hand it that many
        poses. Tiling rather than zero-padding: an all-zero trajectory is a real
        pose sequence sitting on the ego, so it would attend somewhere specific and
        cost the same compute while pulling the attention pattern towards a place
        no candidate goes. Padding rows are sliced off before anything reads them.

        Note that scoring is weakly context-dependent, so what fills the padding is
        not entirely inert. `Bev_refiner`'s temporal self-attention lets query slots
        attend to each other, which means a candidate's logits depend a little on
        which trajectories occupy the other slots. Measured on a CARE window with
        the pretrained checkpoint: scoring 5 candidates padded to 64 differs from
        scoring the same 5 inside the model's own full proposal set by 0.0037 in
        predicted PDMS, against a spread across proposals of 0.49-0.88. Small, but
        it is why `forward_train` always hands the refiner the same number of slots
        rather than sizing them to the batch -- the alternative is a scorer whose
        output moves with the batch composition.
        """
        num_candidates = candidates.shape[1]
        if num_candidates == self.num_proposals:
            return candidates
        repeats = -(-self.num_proposals // num_candidates)  # ceil
        return candidates.repeat(1, repeats, 1, 1)[:, : self.num_proposals]

    def score_candidates(
        self, features: Dict[str, torch.Tensor], candidates: torch.Tensor
    ) -> torch.Tensor:
        """Sub-score logits for trajectories the model did not necessarily propose.

        :param candidates: (B, K, P, 3) in the ego frame, K <= num_proposals
        :return: (B, K, 6) logits, in rl.scoring.SCORE_KEYS order

        Inference-only convenience; a training step wants `forward_train`, which
        returns the proposals from the same encode.
        """
        return self.forward_train(features, candidates)[1]
