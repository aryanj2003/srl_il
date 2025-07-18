# srl_il/srl_il/dataset/custom_sequence_dataset.py

from srl_il.dataset.dataset_base import SequenceDataset, get_train_val_test_seq_datasets
from srl_il.dataset.implement_dataset_base import CustomTrajectoryDataset


class CustomSequenceDataset(SequenceDataset):
    def __init__(
        self,
        trajectory_dataset: CustomTrajectoryDataset,
        window_size: int = 21,
        keys_traj_cfg: list = None,
        keys_global_cfg: list = None,
        pad_before: bool = False,
        pad_after: bool = False,
        pad_type: str = 'near',
        **kwargs,
    ):
        """
        Args:
            trajectory_dataset:  your CustomTrajectoryDataset instance
            window_size (int):   how many timesteps per slice
            keys_traj_cfg (list):  the [[name, src_name, start, end], …] spec
            keys_global_cfg (list): same idea for any global keys
        """
        self.keys_traj_cfg = keys_traj_cfg or []
        self.keys_global_cfg = keys_global_cfg or []
        super().__init__(
            trajectory_dataset,
            window_size=window_size,
            keys_traj=self.keys_traj_cfg,
            keys_global=self.keys_global_cfg,
            pad_before=pad_before,
            pad_after=pad_after,
            pad_type=pad_type,
            **kwargs,
        )


class CustomSequenceTrainValTest:
    """
    Wraps your CustomSequenceDataset and splits into train/val/test
    so that Hydra can instantiate it under dataset_cfg.data._target_.
    """
    def __init__(
        self,
        data_path: str,
        keys_traj: list,
        keys_global: list,
        preloaded: bool,
        test_fraction: float,
        val_fraction: float,
        window_size_train: int,
        window_size_test: int,
        pad_before: bool,
        pad_after: bool,
        pad_type: str,
        random_seed: int,
        num_worker: int = None,
        **kwargs,
    ):
        # 1) load raw trajectories from YAML
        traj_ds = CustomTrajectoryDataset(
            data_path=data_path,
            keys_traj=keys_traj,
            data_type="train",
        )

        print(f"keys_traj: {keys_traj}")
        print(f"keys_global: {keys_global}")


        # 2) wrap in sliding‐window sequence dataset
        seq_ds = CustomSequenceDataset(
            trajectory_dataset=traj_ds,
            window_size=window_size_train,
            keys_traj_cfg=keys_traj,
            keys_global_cfg=keys_global,
            pad_before=pad_before,
            pad_after=pad_after,
            pad_type=pad_type,
        )

        # Debugging: Check the number of sequences in the dataset
        print(f"Total sequences in the dataset: {len(seq_ds)}")

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

        # Debugging: Check the number of items in train/val/test
        print(f"Train data: {len(self.train_data)} sequences")
        print(f"Validation data: {len(self.val_data)} sequences")
        print(f"Test data: {len(self.test_data)} sequences")
