import abc
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence
import h5py
import numpy as np
import torch
import torch.utils
from torch.utils.data import Dataset, Subset, Dataset
import pathlib
import struct
import torch.nn.functional as F
from glob import glob
import pdb
import logging, os


class TrajectoryDataset(Dataset, abc.ABC):
    """
    A dataset containing trajectories. Each data sample is a trajectory.
    TrajectoryDataset[i] returns two dicts. 
        traj_dict:
            Key: the name of the data. e.g. camera, action, etc
            Value: The data. The first dimention is the full trajectory length.
        global_dict:
            Key: the name of the data. e.g. goal camera, natural language description, etc
            Value: The data. The first dimention is the full trajectory length.
    The data in these dicts can be hdf5 pointers(the hdf5.Dataset object that hasn't been loaded into memory) or numpy arrays.
    The actual data loading is done in `load` method.
    """

    @abc.abstractmethod
    def get_seq_length(self, idx):
        """
        Returns the length of the idx-th trajectory.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def __len__(self):
        """
        Returns the number of trajectories.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    def load(self, data, key, is_global=False):
        """
        Load the data into memory and convert to torch array.
        args: 
        data: The data to be loaded, should be the same type as the elements returned by __getitem__
        key: The key of the data to be loaded
        is_global: If True, the data is global data. Otherwise, the data is trajectory data.
        """
        raise NotImplementedError


class TrajectorySubset(TrajectoryDataset, Subset):
    """
    Subset of a trajectory dataset at specified indices.

    Args:
        dataset (TrajectoryDataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """

    def __init__(self, dataset: TrajectoryDataset, indices: Sequence[int]):
        Subset.__init__(self, dataset, indices)

    def get_seq_length(self, idx):
        return self.dataset.get_seq_length(self.indices[idx])

    def __len__(self):
        return Subset.__len__(self)
    
    def load(self, data, key, is_global=False):
        return self.dataset.load(data, key, is_global)


def random_split_trajectory_dataset(dataset:TrajectoryDataset, N_elems:Sequence[int], random_seed:int=42):
    Ntotal = sum(N_elems)
    assert len(dataset) == Ntotal
    indices = torch.randperm(Ntotal).tolist()
    subsets = []
    start = 0
    for length in N_elems:
        end = start + length
        subsets.append(TrajectorySubset(dataset, indices[start:end]))
        start = end
    return subsets

class SequenceDataset(Dataset):
    """
    Slices trajectories from a TrajectoryDataset into windows of a fixed length.
    TrajectoryDataset[i] returns three tuples. 
        traj_tuple:
            Tuple of the trajectory data. Each element is a tensor of shape [window_size, ...]
        traj_masks:
            Tuple of the valid masks. True means valid, False means padded. Each element is a tensor of shape [window_size].
        global_tuple:
            Tuple of the global data. Each element is a tensor

    Args:
        keys_traj: list of tuples: each element represents a key in the dataset
            [keyname, srcname, start, end] The start and end are the indices of the window. Start and end can be none for the full sequence.
        keys_global: list of strings: each element represents a key in the global dataset
        pad_before: If True, pads the sequence before the start index.
        pad_after: If True, pads the sequence after the end index.
        pad_type: The type of padding. Can be 'zero', 'near'. Default is 'zero'.
    """
    def __init__(
        self,
        dataset: TrajectoryDataset,
        window_size: int,
        keys_traj: Sequence[tuple[str, str, Optional[int], Optional[int]]],
        keys_global: Sequence[str],
        pad_before: False,
        pad_after: True,
        pad_type: str = "zero",
    ):
        self._dataset = dataset
        self._window_size = window_size
        print(f"[SequenceDataset] START init: #traj={len(dataset)}, window_size={window_size}")
        self._keys_traj = keys_traj
        self._keys_global = keys_global
        self._pad_before = pad_before
        self._pad_after = pad_after
        self._pad_type = pad_type



        # [start: end] will be loaded from the dataset[idx]
        self._idx_to_slice = []  # list of tuples: (row_indx, start, end, pad_before, pad_after)

        seq_len = len(self.dataset)  # total time steps
        for j in range(-window_size+1, seq_len):  # logical window start
            start = max(0, j)
            end = min(seq_len, j + window_size)
            pad_before = max(0, -j)
            pad_after  = max(0, j + window_size - seq_len)

            if (not self._pad_before) and pad_before > 0:
                continue
            if (not self._pad_after) and pad_after > 0:
                continue

            # row_indx fixed at 0 because your dataset indexes time, not multiple trajs
            self._idx_to_slice.append((0, start, end, pad_before, pad_after))


        print(f"[SequenceDataset] DONE slicing: total windows={len(self._idx_to_slice)}")

        # check the keys
        data_sample_traj, data_sample_global = self.dataset[0]
        all_names = []
        for key, src, start, end in self._keys_traj:
            assert src in data_sample_traj, f"Key {key} is from {src}, which is not found in the dataset"
            start = 0 if start is None else start
            end = self._window_size if end is None else end
            assert 0<= start <= end <= self._window_size, "start must be >= 0"
            all_names.append(key)
        for key in self._keys_global:
            assert key in data_sample_global, f"Key {key} not found in the dataset"
            all_names.append(key)
        assert len(all_names) == len(set(all_names)), f"Duplicate keys found in {all_names}"
        assert self._pad_type in ["zero", "near"], f"Unknown pad_type {self._pad_type}"


    @property
    def dataset(self):
        return self._dataset
    
    @property
    def window_size(self):
        return self._window_size
    
    @property
    def keys_traj(self):
        return self._keys_traj
    
    @property
    def keys_global(self):
        return self._keys_global
    
    @property
    def pad_before(self):
        return self._pad_before
    
    @property
    def pad_after(self):
        return self._pad_after
    
    @property
    def pad_type(self):
        return self._pad_type
    

    def get_seq_length(self, idx: int) -> int:
        return self._window_size

    def __len__(self):
        return len(self._idx_to_slice)
    

    def __getitem__(self, idx):
        # setup logging
        output_dir = "/home/aryan/IL_Workspace/srl_il/output"
        os.makedirs(output_dir, exist_ok=True)
        logging.basicConfig(
            filename=f"{output_dir}/sequence_getitem_debug.log",
            level=logging.DEBUG,
            format="%(asctime)s [DEBUG] %(message)s"
        )

        # Which index to get and window size
        row_indx, start, end, pad_before, pad_after = self._idx_to_slice[idx]
        traj_masks = {}
        ret_dict   = {}

        # Pull out one “trajectory” sample
        step_i, global_i = self.dataset[row_indx]

        # 1) Global keys stay identical
        for key in self._keys_global:
            ret_dict[key] = self.dataset.load(global_i[key], key, is_global=True)

        # 2) Per-timestep keys
        expected_cols = {
            'gripper_pressure': 3,
            'joint_states_fixed': 24,
            'force_torque_sensor_broadcaster_wrench': 6,
            'servo_node_delta_twist_cmds': 6
        }
        W = self._window_size

        for key, src, *_ in self._keys_traj:
            print(f"[DBG] idx={idx}, traj_i={row_indx}, key={key!r}, src={src!r}, "
          f"start={start}, end={end}, "
          f"type(data_traj[src])={type(step_i[src])}, "
          f"value={step_i[src][start:end]!r}")
            D = expected_cols[src]
            traj_parts = []
            mask_parts = []
            for w in range(W):
                t = start + (w - pad_before)
                if 0 <= t < len(self._dataset):
                    row = self._dataset[t][0]  
                    if (src in row) and (row[src] is not None):
                        data = row[src]

                        # normalize data to a 1D list of length D
                        if isinstance(data, torch.Tensor):
                            data = data.detach().cpu().view(-1).tolist()
                        elif isinstance(data, (list, tuple, np.ndarray)):
                            data = np.asarray(data).reshape(-1).tolist()
                        else:
                            print(f"[SEQ] {src}@t={t}: unsupported type {type(data)}; marking invalid")
                            part = torch.zeros(D)
                            mask_val = False
                            traj_parts.append(part); mask_parts.append(mask_val)
                            continue

                        if len(data) != D:
                            print(f"[SEQ] {src}@t={t}: length {len(data)} != expected {D}; marking invalid")
                            part = torch.zeros(D)
                            mask_val = False
                        else:
                            part = torch.tensor(data)
                            # invalid if any non-finite (NaN/±Inf)
                            if not torch.isfinite(part).all():
                                print(f"[SEQ] {src}@t={t}: non-finite values detected; marking invalid")
                                part = torch.zeros(D)
                                mask_val = False
                            else:
                                mask_val = True
                    else:
                        # key missing for this timestep → invalid (no fake zeros counted as real)
                        print(f"[SEQ] {src}@t={t}: missing; mask=False")
                        part = torch.zeros(D)
                        mask_val = False
                else:
                    part = torch.zeros(D)
                    mask_val = False

                traj_parts.append(part)
                mask_parts.append(torch.tensor(mask_val, dtype=torch.bool))

            # → [W, D]
            traj_tensor = torch.stack(traj_parts, dim=0)
            # → [W]
            traj_mask   = torch.stack(mask_parts, dim=0)

            ret_dict[key]   = traj_tensor
            traj_masks[key] = traj_mask

        

        # --- robust aliasing for actions (maps by SOURCE name) ---
        # If your keys_traj maps ('actions_cleaned', 'servo_node_delta_twist_cmds', ...),
        # this will alias actions <- actions_cleaned safely.
        src_to_dest = {src: key for (key, src, *_) in self._keys_traj}
        twist_dest = src_to_dest.get('servo_node_delta_twist_cmds')

        if twist_dest and twist_dest in ret_dict:
            ret_dict['actions']   = ret_dict[twist_dest]
            traj_masks['actions'] = traj_masks[twist_dest]
        elif 'servo_node_delta_twist_cmds' in ret_dict:
            # fallback if dest == src
            ret_dict['actions']   = ret_dict['servo_node_delta_twist_cmds']
            traj_masks['actions'] = traj_masks['servo_node_delta_twist_cmds']
        else:
            # last-resort safety (prevents crashes if config changes)
            W, D = self._window_size, 6
            ret_dict['actions']   = torch.zeros(W, D, dtype=torch.float32)
            traj_masks['actions'] = torch.zeros(W, dtype=torch.bool)



        # joint_states_fixed is [W, 24] in order: position(12) + velocity(6) + effort(6)
        js = ret_dict['joint_states_fixed']          # [W,24]
        ret_dict['joint_states_position'] = js[..., :12]
        ret_dict['joint_states_velocity'] = js[..., 12:18]
        ret_dict['joint_states_effort']   = js[..., 18:24]

        # re-use the same mask
        traj_masks['joint_states_position'] = traj_masks['joint_states_fixed']
        traj_masks['joint_states_velocity'] = traj_masks['joint_states_fixed']
        traj_masks['joint_states_effort']   = traj_masks['joint_states_fixed']
        return ret_dict, traj_masks



def get_train_val_test_seq_datasets(
    traj_dataset: TrajectoryDataset,
    test_fraction: float,
    val_fraction:float,
    window_size_train: int,
    window_size_test: int,
    keys_traj: Sequence[tuple[str, Optional[int], Optional[int]]],
    keys_global: Sequence[str],
    pad_before: bool,
    pad_after: bool,
    pad_type: str,
    random_seed: int = 42,
):
    """
    Splits a TrajectoryDataset into train, validation, and test sets. And build the SequenceDataset for each set.
    The definition of the train, val, test are different from the standard split.
        Train set is used for training, 
        Validation set is sampled from the same trajectories as the train set
        Test set is sampled from the remaining trajectories.
    """
    N_trajs = len(traj_dataset)
    if test_fraction == 0:
        train_traj_dataset = traj_dataset
        test_traj_dataset = None
    else:
        N_test = max(1, int(test_fraction * N_trajs))
        train_traj_dataset, test_traj_dataset = random_split_trajectory_dataset(
            traj_dataset, 
            [N_trajs - N_test, N_test], 
            random_seed=random_seed
        )

    seq_ds_kwargs = {
        "keys_traj": keys_traj,
        "keys_global": keys_global,
        "pad_before": pad_before,
        "pad_after": pad_after,
        "pad_type": pad_type,
    }
    train_seq_ds = SequenceDataset(train_traj_dataset, window_size=window_size_train, **seq_ds_kwargs)

    N_train = len(train_seq_ds)
    N_val = int(val_fraction * N_train)
    train_ds, val_ds = torch.utils.data.random_split(
        train_seq_ds, 
        [N_train - N_val, N_val], 
        generator= torch.Generator().manual_seed(random_seed)
    )

    if test_fraction == 0:
        test_ds = None
    else:
        test_ds = SequenceDataset(test_traj_dataset, window_size=window_size_test,  **seq_ds_kwargs)
    return train_ds, val_ds, test_ds
