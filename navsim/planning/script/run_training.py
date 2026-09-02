import navsim._torch_pytree_compat  # noqa: F401 -- must run before transformers is imported anywhere, even transitively

import os
from typing import Tuple
from pathlib import Path
import logging

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import CacheOnlyDataset, Dataset, WaymoCacheOnlyDataset
from navsim.planning.training.agent_lightning_module import AgentLightningModule
from pytorch_lightning.loggers import WandbLogger

import random
from torch.utils.data import Subset
from torch.utils.data import ConcatDataset
logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs

    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )

    return train_data, val_data


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training an agent.
    :param cfg: omegaconf dictionary
    """

    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    # Pin this rank's GPU BEFORE the agent is built. rap_agent.py:115-116 does
    #     device_id = torch.cuda.current_device(); self.device = torch.device(f"cuda:{device_id}")
    # and then self.to(self.device) -- but the agent is constructed here, before
    # trainer.fit(), which is where Lightning normally assigns each rank its device. So
    # torch.cuda.current_device() is still 0 in every subprocess and all N ranks load the
    # model onto cuda:0. With 8 ranks that OOMs a 24 GB card during construction (observed,
    # job 7410887: eight processes on GPU 0, 3.31 MiB free).
    # Lightning's SubprocessScriptLauncher sets LOCAL_RANK for every child
    # (subprocess_script.py:125,130), so it is available this early.
    if torch.cuda.is_available():
        _local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if _local_rank < torch.cuda.device_count():
            torch.cuda.set_device(_local_rank)
        logger.info(f"Rank {_local_rank} pinned to cuda:{torch.cuda.current_device()}")

    # RAP_MEM_DEBUG=1 prints a per-rank CUDA memory breakdown at the points that separate
    # "resident" cost (weights, DDP gradient buckets, optimizer state) from "per-step" cost
    # (activations). Batch size only moves the second number, so if fit-start is already
    # near capacity the batch size is the wrong lever.
    if os.environ.get("RAP_MEM_DEBUG"):
        class _MemProbe(pl.Callback):
            @staticmethod
            def _report(tag: str) -> None:
                rank = int(os.environ.get("LOCAL_RANK", 0))
                alloc = torch.cuda.memory_allocated() / 2**30
                reserved = torch.cuda.memory_reserved() / 2**30
                peak = torch.cuda.max_memory_allocated() / 2**30
                logger.info(
                    f"[mem][rank {rank}] {tag}: allocated {alloc:.2f} GiB, "
                    f"reserved {reserved:.2f} GiB, peak {peak:.2f} GiB"
                )

            def on_fit_start(self, trainer, pl_module):
                # Everything resident before a single activation exists.
                self._report("fit start (weights + ddp buckets + optim)")

            def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
                if batch_idx < 3:
                    self._report(f"train batch {batch_idx} start")

            def on_before_backward(self, trainer, pl_module, loss):
                if trainer.global_step < 3:
                    self._report(f"step {trainer.global_step} after forward")

            def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                if batch_idx < 3:
                    self._report(f"train batch {batch_idx} end")
                    torch.cuda.reset_peak_memory_stats()

        _mem_callbacks = [_MemProbe()]
    else:
        _mem_callbacks = []

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent)

    logger.info("Building Lightning Module")
    lightning_module = AgentLightningModule(
        agent=agent,
    )

    if cfg.use_cache_without_dataset:
        logger.info("Using cached data without building SceneLoader")
        assert (
            not cfg.force_cache_computation
        ), "force_cache_computation must be False when using cached data without building SceneLoader"
        assert (
            cfg.cache_path is not None
        ), "cache_path must be provided when using cached data without building SceneLoader"

        cached_logs = [log_name.name.replace(".pkl", "") for log_name in Path(cfg.cache_path).iterdir()]
        train_logs = [log_name for log_name in cached_logs if log_name not in cfg.val_logs]
        val_logs = [log_name for log_name in cached_logs if log_name in cfg.val_logs]

        if 'waymo' in cfg.dataset['_target_']:
            train_data = WaymoCacheOnlyDataset(
                cache_path=cfg.cache_path,
                split='training'
            )
            val_data = WaymoCacheOnlyDataset(
                cache_path=cfg.cache_path,
                split='val',
            )
            # # split val_data by 80/20
            # import random
            # from torch.utils.data import ConcatDataset, Subset
            # N = len(val_data)
            # indices = random.sample(range(N), int(0.8*N))
            # the_rest = [i for i in range(N) if i not in indices]
            # train_data = Subset(val_data, indices)
            # val_data = Subset(val_data, the_rest)
        else:
            train_data = CacheOnlyDataset(
                cache_path=cfg.cache_path,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders(),
            log_names=train_logs,
        )
            val_data = CacheOnlyDataset(
                cache_path=cfg.cache_path,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders(),
                log_names=val_logs,
                split='val'
            )

            train_data_perturbed = CacheOnlyDataset(
                cache_path=cfg.cache_path_perturbed,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders())
            N = len(train_data_perturbed)
            # Fraction of the perturbed pool to draw. Default 0.1 matches the released
            # recipe, which caches the full split; set to 1.0 when the cache is already
            # sized to what should be consumed (see make_cache_subsets.py).
            indices = random.sample(range(N), int(cfg.get('perturbed_fraction', 0.1)*N))
            print(f'len(perturbed): {len(indices)}')
            train_data_perturbed = Subset(train_data_perturbed, indices)

            train_data_others = CacheOnlyDataset(
                cache_path=cfg.cache_path_others,
                feature_builders=agent.get_feature_builders(),
                target_builders=agent.get_target_builders())
                
            train_data_others.score_mask=False
            N = len(train_data_others)
            indices = random.sample(range(N), int(cfg.get('others_fraction', 0.05)*N))
            print(f'len(others): {len(indices)}')
            train_data_others = Subset(train_data_others, indices)

            train_data = ConcatDataset([train_data, train_data_perturbed, train_data_others])

    else:
        logger.info("Building SceneLoader")
        train_data, val_data = build_datasets(cfg, agent)

    logger.info("Building Datasets")
    train_dataloader = DataLoader(train_data, **cfg.dataloader.params, shuffle=True)
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(val_data, **cfg.dataloader.params, shuffle=False)
    logger.info("Num validation samples: %d", len(val_data))

    logger.info("Building Trainer")
    # save_dir keeps wandb's own files next to the run instead of in the cwd; the
    # checkpoint path is pinned separately via get_training_callbacks(cfg.output_dir),
    # which short-circuits ModelCheckpoint's logger-derived fallback entirely.
    trainer = pl.Trainer(**cfg.trainer.params, callbacks=agent.get_training_callbacks(cfg.output_dir) + _mem_callbacks, logger=WandbLogger(project="rap", name=cfg.experiment_name, id=cfg.experiment_name, save_dir=cfg.output_dir),
            )

    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        ckpt_path='last'
    )


if __name__ == "__main__":
    main()
