"""Scenes, from either source, behind one interface.

A round needs four things from a scene, and needs them the same way whether the
scene is a CARE clip window or a navtrain token:

    features(token)          the model input tensors
    human_trajectory(token)  the recorded ego future, ego frame
    scoring_payload(token)   whatever rl.scoring's backend needs to label it
    tokens                   which scenes exist

The two sources differ in exactly one way that leaks past this interface, and it is
worth stating plainly rather than discovering in a stack trace: **they have
different camera counts**. A navtrain sample carries 4 cameras, a CARE window
carries 1 (`generate_ras_logs.py` writes CAM_F0 and nothing else). Tensors of
(4, 3, 448, 768) and (1, 3, 448, 768) do not stack, so a batch is always drawn from
one source or the other -- never mixed. `iterate_batches` below is the only place
that has to know it, and rl/train.py alternates between the two per step.

`camera_feature`, not `rendered_camera_feature`: RAPConfig sets
`distill_feature=False`, so `AgentLightningModule._step` runs and the swap to the
rasterized view in `_step_distill` does not. The checkpoint was trained on
photographs, and a CARE clip's photographs are the extracted video frames, so both
sources feed the same field for the same reason.
"""

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple
import gzip
import hashlib
import json
import pickle

import numpy as np
import torch


def split_tokens(tokens: Sequence[str], val_fraction: float) -> Tuple[List[str], List[str]]:
    """Deterministic train/val split keyed on the token hash.

    Hashing rather than shuffling with a seed means collect.py, train.py and
    eval.py agree on the split without any of them writing a file, and the split
    survives the scene index being rebuilt in a different order -- which it will be
    if the CARE export gains a clip.
    """
    train, val = [], []
    threshold = val_fraction * 2 ** 32
    for token in tokens:
        bucket = int.from_bytes(hashlib.md5(token.encode()).digest()[:4], "little")
        (val if bucket < threshold else train).append(token)
    return train, val


# ------------------------------------------------------------------------- sources


class CareSource:
    """CARE clip windows. Features are built from images on demand."""

    name = "care"

    def __init__(self, config):
        from navsim.agents.rap_dino.rap_features import RAPFeatureBuilder
        from navsim.common.dataclasses import SensorConfig
        from rl.care import CareClips
        from rl.model import build_rap_config

        self.config = config
        self.clips = CareClips(
            config.care_root,
            num_history_frames=config.num_history_frames,
            num_future_frames=config.num_future_frames,
            stride=config.care_frame_stride,
        )
        self._by_token = {scene.token: scene for scene in self.clips.scenes}
        self._feature_builder = RAPFeatureBuilder(build_rap_config(config))
        # Matches RAPAgent.get_sensor_config: the fourth history frame only, four
        # camera slots of which CARE fills one.
        self._sensor_config = SensorConfig(
            cam_f0=[3], cam_l0=[3], cam_l1=[], cam_l2=[],
            cam_r0=[3], cam_r1=[], cam_r2=[], cam_b0=[3], lidar_pc=[],
        )

    @property
    def tokens(self) -> List[str]:
        return self.clips.tokens()

    def features(self, token: str) -> Dict[str, torch.Tensor]:
        return self.clips.features(
            self._by_token[token], self._feature_builder, self._sensor_config
        )

    def human_trajectory(self, token: str) -> np.ndarray:
        return self.clips.human_trajectory(self._by_token[token])

    def scoring_payload(self, token: str):
        """`(human_trajectory, agent_corners, label_drivable)` for `score_care`."""
        scene = self._by_token[token]
        return (
            self.clips.human_trajectory(scene),
            self.clips.agent_corners(scene),
            self.config.care_label_drivable,
        )


class NavtrainSource:
    """navtrain scenes, read straight out of the supervised feature cache.

    `CacheOnlyDataset` is bypassed on purpose. It walks every log directory on
    construction to build its token index, which on a 14,684-token cache over NFS
    is minutes -- paid once per process, and a round runs several. The layout it
    walks is `<cache>/<log>/<token>/{rap_feature,rap_target}.gz`, which is what this
    walks instead, memoising the result into the run directory. The cache itself is
    never written to: it belongs to the supervised pipeline.
    """

    name = "navtrain"

    def __init__(self, config):
        self.config = config
        self.cache_path = Path(config.navtrain_cache_path)
        self._index = self._load_or_build_index()
        self._metric_cache_paths: Optional[Dict[str, str]] = None

    def _load_or_build_index(self) -> Dict[str, str]:
        index_path = Path(self.config.output_path) / "navtrain_index.json"
        if index_path.is_file():
            with open(index_path) as handle:
                return json.load(handle)

        index: Dict[str, str] = {}
        for log_dir in sorted(self.cache_path.iterdir()):
            if not log_dir.is_dir():
                continue
            for token_dir in sorted(log_dir.iterdir()):
                if (token_dir / "rap_feature.gz").is_file() and (
                    token_dir / "rap_target.gz"
                ).is_file():
                    index[token_dir.name] = str(token_dir)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, "w") as handle:
            json.dump(index, handle)
        print(f"[data] indexed {len(index)} navtrain tokens -> {index_path}")
        return index

    @property
    def tokens(self) -> List[str]:
        # sorted(), not dict order: the index is JSON round-tripped and a round's
        # seeded subsample has to land on the same scenes on every machine.
        return sorted(self._index)

    def _read(self, token: str, which: str) -> Dict:
        with gzip.open(Path(self._index[token]) / f"{which}.gz", "rb") as handle:
            return pickle.load(handle)

    def features(self, token: str) -> Dict[str, torch.Tensor]:
        return self._read(token, "rap_feature")

    def human_trajectory(self, token: str) -> np.ndarray:
        return self._read(token, "rap_target")["trajectory"].numpy().astype(np.float32)

    def scoring_payload(self, token: str) -> str:
        if self._metric_cache_paths is None:
            from navsim.common.dataloader import MetricCacheLoader

            self._metric_cache_paths = MetricCacheLoader(
                Path(self.config.metric_cache_path)
            ).metric_cache_paths
        return str(self._metric_cache_paths[token])

    def scorable_tokens(self) -> List[str]:
        """Tokens that have both cached features and a metric cache.

        The metric cache is a superset built for another project, so in practice
        this drops nothing -- but a partial regeneration of either side would
        otherwise surface as every trajectory scoring NaN.
        """
        if self._metric_cache_paths is None:
            self.scoring_payload(self.tokens[0])
        return [token for token in self.tokens if token in self._metric_cache_paths]


def build_sources(config, names: Sequence[str] = ("care", "navtrain")) -> Dict[str, object]:
    """Construct the requested sources. `regular_ratio=0` drops navtrain entirely."""
    sources: Dict[str, object] = {}
    if "care" in names:
        sources["care"] = CareSource(config)
    if "navtrain" in names and config.regular_ratio > 0:
        sources["navtrain"] = NavtrainSource(config)
    return sources


# ------------------------------------------------------------------------ batching


class SceneDataset(torch.utils.data.Dataset):
    """Features for a fixed list of tokens from one source.

    One source per dataset, because the camera counts differ -- see the module
    docstring.
    """

    def __init__(self, source, tokens: Sequence[str], include_human: bool = False):
        self.source = source
        self.tokens = list(tokens)
        # Read in the worker process, not the training loop: for navtrain this is
        # one small gzip per sample off NFS, and doing it inline would serialise it
        # behind the forward pass.
        self.include_human = include_human

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int):
        token = self.tokens[index]
        features = self.source.features(token)
        if self.include_human:
            features = dict(features)
            features["human_trajectory"] = torch.as_tensor(
                np.asarray(self.source.human_trajectory(token), dtype=np.float32)
            )
        return token, features


def collate(samples):
    """Stack feature tensors, keep tokens as a list.

    `token` is a str and would break `default_collate`'s tensor path; the
    zero-dimensional `camera_valid` stacks fine and is carried through because
    `RAPModel` reads it off the features dict.
    """
    tokens = [sample[0] for sample in samples]
    features = {
        key: torch.stack([sample[1][key] for sample in samples])
        for key in samples[0][1]
        if torch.is_tensor(samples[0][1][key])
    }
    return tokens, features


def loader(
    source,
    tokens: Sequence[str],
    batch_size: int,
    num_workers: int,
    shuffle: bool = False,
    include_human: bool = False,
):
    return torch.utils.data.DataLoader(
        SceneDataset(source, tokens, include_human),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        drop_last=False,
    )


def iterate_batches(loaders: Dict[str, torch.utils.data.DataLoader], seed: int = 0) -> Iterator:
    """Interleave per-source loaders, yielding `(source_name, tokens, features)`.

    Round-robin weighted by how much each loader has left, so the two sources stay
    mixed through the epoch rather than the smaller one finishing in the first
    tenth of it. A step still only ever sees one source, which is what the camera
    counts require.
    """
    iterators = {name: iter(value) for name, value in loaders.items()}
    remaining = {name: len(value) for name, value in loaders.items()}
    rng = np.random.default_rng(seed)

    while any(count > 0 for count in remaining.values()):
        names = [name for name, count in remaining.items() if count > 0]
        weights = np.array([remaining[name] for name in names], dtype=np.float64)
        name = names[int(rng.choice(len(names), p=weights / weights.sum()))]
        remaining[name] -= 1
        tokens, features = next(iterators[name])
        yield name, tokens, features
