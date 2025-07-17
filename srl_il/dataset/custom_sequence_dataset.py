# srl_il/srl_il/dataset/custom_sequence_dataset.py


from srl_il.dataset.robomimic_dataset import RobomimicTrajectorySequenceDataset
from srl_il.dataset.dataset_base import get_train_val_test_seq_datasets
from srl_il.dataset.implement_dataset_base import CustomTrajectoryDataset


class CustomSequenceDataset(RobomimicTrajectorySequenceDataset):
    def __init__(self, trajectory_dataset, window_size=21, keys_traj_cfg=None,
                 keys_global_cfg=None, **kwargs):
        """
        Args:
            trajectory_dataset:  your CustomTrajectoryDataset instance
            window_size (int):   how many timesteps per slice
            keys_traj_cfg (list):  the [[name, src_name, start, end], …] spec
            keys_global_cfg (list): same idea for any global keys
        """
        # 1) store your configs
        self.keys_traj_cfg   = keys_traj_cfg   or []
        self.keys_global_cfg = keys_global_cfg or []
        # 2) let the parent set itself up
        super().__init__(
            trajectory_dataset,
            window_size=window_size,
            keys_traj=self.keys_traj_cfg,
            keys_global=self.keys_global_cfg,
            **kwargs
        )





class CustomSequenceTrainValTest:
    """
    Wraps your CustomSequenceDataset and splits into train/val/test
    so that Hydra can instantiate it under dataset_cfg.data._target_.
    """
    def __init__(
        self,
        # these names must match what you put in the YAML
        data_path,
        keys_traj,
        keys_global,
        preloaded,
        test_fraction,
        val_fraction,
        window_size_train,
        window_size_test,
        pad_before,
        pad_after,
        pad_type,
        random_seed,
        num_worker=None,       # if you need it
        **kwargs,              # catch any extras
    ):
        # 1) raw trajectories
        traj_ds = CustomTrajectoryDataset(
            data_path=data_path,
            keys_traj=keys_traj,
            data_type="train",
        )

        # 2) sliding‐window sequence
        seq_ds = CustomSequenceDataset(
            trajectory_dataset=traj_ds,
            window_size=window_size_train,
            keys_traj_cfg=keys_traj,
            keys_global_cfg=keys_global,
        )

        # 3) split into train/val/test
        self.train_data, self.val_data, self.test_data = get_train_val_test_seq_datasets(
            seq_ds,
            test_fraction=test_fraction,
            val_fraction=val_fraction,
            window_size_train=window_size_train,
            window_size_test=window_size_test,
            keys_traj=keys_traj,
            keys_global=keys_global,
            pad_before=pad_before,
            pad_after=pad_after,
            pad_type=pad_type,
            random_seed=random_seed,
        )

    # Hydra will look for attributes .train_data, .val_data, .test_data
