import datetime
import numpy as np
from pathlib import Path
import os
import hydra
from copy import deepcopy
from typing import Any, Dict, List, Optional
from ..algo.base_algo import Algo, TrainerMixin
from .pipeline_base import Pipeline, AlgoMixin, WandbMixin, DatasetMixin, Lr_SchedulerMixin, NormalizationMixin
from .vis_mixins.env_rollout_mixin import SimulationEnvMixin
from .vis_mixins.visualize_projected_actions import ActionProjectorMixin
from .data_augmentation import DataAugmentationMixin
from srl_il.dataset.custom_sequence_dataset import CustomSequenceDataset
from srl_il.dataset.implement_dataset_base import CustomTrajectoryDataset
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import json
import logging


class ImitationLearningPipeline(Pipeline, AlgoMixin, DatasetMixin, Lr_SchedulerMixin, WandbMixin, DataAugmentationMixin, NormalizationMixin, SimulationEnvMixin, ActionProjectorMixin):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(self.algo, Algo), "algo should be an instance of Algo"
        assert isinstance(self.algo, TrainerMixin), "algo should be an instance of TrainerMixin"
        if self.resume and self.resume_path is not None:
            self.load_checkpoint(self.resume_path)

    def _init_workspace(self, **cfg):
        super()._init_workspace(**cfg)
        
        self.training_config = cfg["training_cfg"]
        # setting up the training
        self.epoch = 1
        self.best_train_loss = 1e10
        self.best_eval_loss = 1e10
        if self.training_config.rollout.enabled:
            os.makedirs(self.training_config.rollout.video.video_dir, exist_ok=True)


        # Set up logging
        logging.basicConfig(
            filename=os.path.join(self.output_dir, "pipeline_init.log"),
            level=logging.DEBUG,
            format="%(asctime)s [DEBUG] %(message)s"
        )

        # Initialize datasets
        logging.debug("Initializing datasets in _init_workspace")
        print(f"[DEBUG] Initializing datasets in _init_workspace")
        data_path = cfg.get("dataset_cfg", {}).get("data_path", "/home/aryan/IL_Workspace/data_collection_output")
        keys_traj = ["gripper_pressure", "servo_node_delta_twist_cmds", "force_torque_sensor_broadcaster_wrench", "joint_states_fixed"]
        keys_traj_cfg = [[key, key, None, None] for key in keys_traj]  # [name, src_name, start, end]
        
        try:
            # Initialize CustomTrajectoryDataset
            trajectory_dataset = CustomTrajectoryDataset(
                data_path=data_path,
                keys_traj=keys_traj,
                data_type="train",
                joint_states_stride=1,
                preloaded=False
            )
            logging.debug("CustomTrajectoryDataset initialized successfully")
            print(f"[DEBUG] CustomTrajectoryDataset initialized successfully")

            # Initialize CustomSequenceDataset
            self.train_dataset = CustomSequenceDataset(
                trajectory_dataset=trajectory_dataset,
                window_size=5,
                keys_traj_cfg=keys_traj_cfg,
                keys_global_cfg=[],
                pad_before=False,
                pad_after=False,
                pad_type='near',
                device=cfg.get("dataset_cfg", {}).get("device", "cpu")
            )
            logging.debug("train_dataset initialized successfully")
            print(f"[DEBUG] train_dataset initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize train_dataset: {e}")
            print(f"[ERROR] Failed to initialize train_dataset: {e}")
            raise

        # Set eval_dataset to None unless specified in config
        self.eval_dataset = None
        eval_data_path = cfg.get("dataset_cfg", {}).get("eval_data_path")
        if eval_data_path:
            try:
                eval_trajectory_dataset = CustomTrajectoryDataset(
                    data_path=eval_data_path,
                    keys_traj=keys_traj,
                    data_type="test",
                    joint_states_stride=1,
                    preloaded=False
                )
                self.eval_dataset = CustomSequenceDataset(
                    trajectory_dataset=eval_trajectory_dataset,
                    window_size=5,
                    keys_traj_cfg=keys_traj_cfg,
                    keys_global_cfg=[],
                    pad_before=False,
                    pad_after=False,
                    pad_type='near',
                    device=cfg.get("dataset_cfg", {}).get("device", "cpu")
                )
                logging.debug("eval_dataset initialized successfully")
                print(f"[DEBUG] eval_dataset initialized successfully")
            except Exception as e:
                logging.warning(f"Failed to initialize eval_dataset: {e}, setting to None")
                print(f"[WARNING] Failed to initialize eval_dataset: {e}, setting to None")
                self.eval_dataset = None

        # Try calling DatasetMixin._init_dataset for additional setup
        try:
            self._init_dataset(**cfg)
            logging.debug("DatasetMixin._init_dataset called successfully")
            print(f"[DEBUG] DatasetMixin._init_dataset called successfully")
        except Exception as e:
            logging.warning(f"Failed to call DatasetMixin._init_dataset: {e}")
            print(f"[WARNING] Failed to call DatasetMixin._init_dataset: {e}")

        # Initialize DataLoaders with custom collate_fn
        try:
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=32,
                shuffle=True,
            )
            logging.debug("train_loader initialized successfully")
            print(f"[DEBUG] train_loader initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize train_loader: {e}")
            print(f"[ERROR] Failed to initialize train_loader: {e}")
            raise

        if self.eval_dataset is not None:
            try:
                self.eval_loader = DataLoader(
                    self.eval_dataset,
                    batch_size=32,
                    shuffle=False,
                    collate_fn=CustomSequenceDataset.collate_fn
                )
                logging.debug("eval_loader initialized successfully")
                print(f"[DEBUG] eval_loader initialized successfully")
            except Exception as e:
                logging.warning(f"Failed to initialize eval_loader: {e}, setting to None")
                print(f"[WARNING] Failed to initialize eval_loader: {e}, setting to None")
                self.eval_loader = None
        else:
            self.eval_loader = None

    def save_checkpoint(self, filepath):
        """Saves a checkpoint alowing to restart training from here
        """
        algo_state = self.algo._get_model_and_optimizer_states()
        checkpoint = {
            'algo_state': algo_state,
            'lr_schedulers': {k: v.state_dict() for k, v in self._lr_schedulers.items()},
            'epoch': self.epoch,
            'best_train_loss': self.best_train_loss,
            'best_eval_loss': self.best_eval_loss,
        }
        torch.save(checkpoint, filepath)

    def save_model(self, filepath):
        """Saves the final model
        """
        checkpoint = self.algo.serialize()
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath):
        """Loads a checkpoint to restart training from a previous point
        """
        checkpoint = torch.load(filepath)
        self.algo._load_model_and_optimizer_states(checkpoint['algo_state'])
        self.epoch = checkpoint['epoch']
        self.best_train_loss = checkpoint['best_train_loss']
        self.best_eval_loss = checkpoint['best_eval_loss']
        for k, v in self._lr_schedulers.items():
            v.load_state_dict(checkpoint['lr_schedulers'][k])

    def run(self):
        """This function is the main training function
        """

        print(
            f"Training params:\n - max_epochs: {self.training_config.num_epochs}\n"
            f" - checkpoint saving every: {self.training_config.steps_saving}\n"
        )
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)

        for epoch in range(1, self.training_config.num_epochs + 1):
            self.epoch = epoch
            metrics = {}
            
            with self._timer.time("train"):
                print("training epoch", epoch)
                self.algo.train_epoch_begin(epoch)
                train_loader_iter = iter(self.train_loader)
                for _ in tqdm(range(self.training_config.num_steps_per_epoch)):
                    with self._timer.time("data_loading"):
                        try:
                            inputs = next(train_loader_iter)
                        except StopIteration:
                            train_loader_iter = iter(self.train_loader)
                            inputs = next(train_loader_iter)
                    
                    batch, mask_batch = inputs
                    batch, mask_batch = self.data_augmentation_train(batch, mask_batch)
                    with self._timer.time("train_step"):
                        self.algo.train_step((batch, mask_batch), epoch)
                    self._step_schedulers(metrics)
                train_metrics = self.algo.train_epoch_end(epoch)
                metrics.update(train_metrics)

            if self.eval_loader is not None:
                with self._timer.time("eval"):
                    print("eval epoch", epoch)
                    self.algo.eval_epoch_begin(epoch)
                    eval_loader_iter = iter(self.eval_loader)
                    for _ in tqdm(range(self.training_config.num_eval_steps_per_epoch)):
                        with self._timer.time("data_loading"):
                            try:
                                inputs = next(eval_loader_iter)
                            except StopIteration:
                                eval_loader_iter = iter(self.eval_loader)
                                inputs = next(eval_loader_iter)
                        batch, mask_batch = inputs
                        batch, mask_batch = self.data_augmentation_eval(batch, mask_batch)
                        with self._timer.time("eval_step"):
                            self.algo.eval_step((batch, mask_batch), epoch)
                    eval_metrics = self.algo.eval_epoch_end(epoch)
                    metrics.update(eval_metrics)
                    if "visualization" in self.training_config.keys() and self.training_config.visualization.enabled and epoch % self.training_config.visualization.every_n_epoch == 0:
                        with self._timer.time("visualization"):
                            num_samples = self.training_config.visualization.num_samples
                            batch, mask_batch = next(iter(self.eval_loader))
                            batch = {k: v[:num_samples].to(self.algo.device) for k, v in batch.items()}
                            mask_batch = {k: v[:num_samples].to(self.algo.device) for k, v in mask_batch.items()}
                            predicted_actions = self.algo.generate(batch, mask_batch)
                            image_paths = self.visualize(batch, predicted_actions, epoch)
                            wandb.log({k: wandb.Video(v) for k, v in image_paths.items()})
                    
            epoch_train_loss = metrics["train_epoch_loss"]
            epoch_eval_loss = metrics.get("eval_epoch_loss", epoch_train_loss)

            ## rollout in env
            if self.training_config.rollout.enabled and epoch % self.training_config.rollout.every_n_epoch == 0:
                with self._timer.time("rollout"):
                    self.algo.set_eval()
                    num_episodes = self.training_config.rollout.num_episodes
                    all_rollout_logs, video_paths = self.rollout_in_sim_env(
                        horizon=self.training_config.rollout.horizon,
                        num_episodes=num_episodes,
                        render=False,
                        video_dir= self.training_config.rollout.video.video_dir,
                        epoch=epoch,
                        video_skip=self.training_config.rollout.video.video_skip,
                        terminate_on_success=self.training_config.rollout.terminate_on_success
                    )
                    print("Rollout logs:")
                    print(json.dumps(all_rollout_logs, indent=4))
                    # summarize results from rollouts to tensorboard and terminal
                    for env_name in all_rollout_logs:
                        rollout_logs = all_rollout_logs[env_name]
                        metrics.update({
                            f"Rollout/{k}/{env_name}": v
                            for k,v in rollout_logs.items()
                        })

                wandb.log({k: wandb.Video(video_path, format="mp4") for k, video_path in video_paths.items()})

            metrics.update({f"timer/{k}":v for k,v in self._timer.analysis().items()})
            wandb.log(metrics)
            
            update_best_model = False
            save_checkpoint = False

            if (
                epoch_eval_loss < self.best_eval_loss
                and self.eval_loader is not None
            ):
                self.best_eval_loss = epoch_eval_loss
                update_best_model = True

            elif (
                epoch_train_loss < self.best_train_loss
                and  self.eval_loader is None
            ):
                self.best_train_loss = epoch_train_loss
                update_best_model = True

            # save checkpoints
            if (
                self.training_config.steps_saving is not None
                and epoch % self.training_config.steps_saving == 0
            ):
                save_checkpoint = True

            if update_best_model:
                self.save_model(os.path.join(self.output_dir, "checkpoints", "best_model.pth"))
                print(f"Best model saved in {self.output_dir}.\n")
            if save_checkpoint:
                self.save_checkpoint(os.path.join(self.output_dir, "checkpoints", f"checkpoint_{epoch}.pth"))
                print(f"Checkpoint saved in {self.output_dir}.\n")