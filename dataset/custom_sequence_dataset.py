# srl_il/srl_il/dataset/custom_sequence_dataset.py
from srl_il.dataset.dataset_base import SequenceDataset, get_train_val_test_seq_datasets
from srl_il.dataset.implement_dataset_base import CustomTrajectoryDataset
import torch
import logging
import os
import numpy as np

class CustomSequenceDataset(SequenceDataset):
    def __init__(
        self,
        trajectory_dataset: CustomTrajectoryDataset,
        window_size: int = 5,
        keys_traj_cfg: list = None,
        keys_global_cfg: list = None,
        pad_before: bool = False,
        pad_after: bool = False,
        pad_type: str = 'near',
        device: str = "cpu",  # Add device parameter with default to cpu
        **kwargs,
    ):
        """
        Args:
            trajectory_dataset:  your CustomTrajectoryDataset instance
            window_size (int):   how many timesteps per slice
            keys_traj_cfg (list):  the [[name, src_name, start, end], …] spec
            keys_global_cfg (list): same idea for any global keys
            device (str):       device to use for tensor operations (e.g., 'cpu' or 'cuda')
        """
        self.trajectory_dataset = trajectory_dataset  # Store the dataset
        self.keys_traj_cfg = keys_traj_cfg or []
        self.keys_global_cfg = keys_global_cfg or []
        self.device = torch.device(device)  # Initialize device

        # print(f"[DEBUG] CustomSequenceDataset.__init__: window_size={self.window_size}")

        print(f"[CustomSequenceDataset] __init__ start: "
              f"traj_len={len(trajectory_dataset)}, "
              f"window_size={window_size}, "
              f"keys_traj_cfg={keys_traj_cfg}, "
              f"device={self.device}")
        super().__init__(
            trajectory_dataset,
            window_size=window_size,
            keys_traj=self.keys_traj_cfg,  # Use keys_traj_cfg directly
            keys_global=self.keys_global_cfg,
            pad_before=pad_before,
            pad_after=pad_after,
            pad_type=pad_type,
            **kwargs,
        )

        print(f"[CustomSequenceDataset] __init__ end: total_sequences={len(self)}, window_size={window_size}")
        
    

    
    def __len__(self):
        return len(self._idx_to_slice)

    def load(self, data, key, is_global=False):
        """
        Delegate loading to CustomTrajectoryDataset's load method.
        """
        logging.debug(f"CustomSequenceDataset.load: key={key}, is_global={is_global}, data={data}, type={type(data)}")
        result = self.dataset.load(data, key, is_global=is_global)
        logging.debug(f"CustomSequenceDataset.load: key={key}, result_shape={result.shape if isinstance(result, torch.Tensor) else 'not a tensor'}")
        return result

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

        logging.debug(f"__getitem__: idx={idx}, row_indx={row_indx}, "
                    f"start={start}, end={end}, pad_before={pad_before}, pad_after={pad_after}")
        print(f"[DEBUG] __getitem__: idx={idx}, row_indx={row_indx}, "
            f"start={start}, end={end}, pad_before={pad_before}, pad_after={pad_after}")

        # Pull out one “trajectory” sample
        step_i, global_i = self.trajectory_dataset[row_indx]

        # 1) Global keys stay identical
        for key in self._keys_global:
            ret_dict[key] = self.dataset.load(global_i[key], key, is_global=True)


        # before the for-loop over keys
        def _expected_dim(src: str) -> int:
            cols = getattr(self.trajectory_dataset, "_colmap", {}).get(src) or []
            return len(cols)
        W = self._window_size


        for key, src, *_ in self._keys_traj:
            print(f"[DBG] idx={idx}, traj_i={row_indx}, key={key!r}, src={src!r}, "
          f"start={start}, end={end}, "
          f"type(data_traj[src])={type(step_i[src])}, "
          f"value={step_i[src][start:end]!r}")
            D = _expected_dim(src)
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
            print(f"[DEBUG] Key={key!r}: tensor_shape={traj_tensor.shape}, mask_shape={traj_mask.shape}")
        print(f"[DEBUG] __getitem__ end: idx={idx}, keys={list(ret_dict.keys())}")




        # --- alias actions to whatever key uses the twist commands ---
        src_to_dest = {src: key for (key, src, *_) in self._keys_traj}
        twist_dest = src_to_dest.get('servo_node_delta_twist_cmds')

        if twist_dest and twist_dest in ret_dict:
            ret_dict['actions']   = ret_dict[twist_dest]
            traj_masks['actions'] = traj_masks[twist_dest]
        elif 'servo_node_delta_twist_cmds' in ret_dict:
            # fallback if you happened to use the same name for dest
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




            

class CustomSequenceTrainValTest:
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
        keys_traj_cfg: list,
        keys_global_cfg: list,
        joint_states_stride: int = 1,
        num_worker: int = None,
        device: str = "cpu",  # Add device parameter
        **kwargs,
    ):
        # 1) load raw trajectories from YAML
        traj_ds = CustomTrajectoryDataset(
            data_path=data_path,
            keys_traj=keys_traj,
            data_type="train",
            joint_states_stride=joint_states_stride,
            preloaded=preloaded,
        )

        print(f"keys_traj: {keys_traj}")
        print(f"keys_global: {keys_global}")

        # Ensure every entry in keys_traj_cfg is (name, src, start, end)
        if not keys_traj_cfg:
            keys_traj_cfg = [[k, k, None, None] for k in keys_traj]

        # 2) build one *full* sliding-window dataset for stats/normalizer
        self.sequence_dataset = CustomSequenceDataset(
            trajectory_dataset=traj_ds,
            window_size=window_size_train,
            keys_traj_cfg=keys_traj_cfg,
            keys_global_cfg=keys_global_cfg,
            pad_before=pad_before,
            pad_after=pad_after,
            pad_type=pad_type,
            device=device,  # Pass device to train/val/test datasets
        )

        # 3) split on the *raw* traj_ds to keep each split small
        self.train_data, self.val_data, self.test_data = get_train_val_test_seq_datasets(
            traj_ds,
            test_fraction=test_fraction,
            val_fraction=val_fraction,
            window_size_train=window_size_train,
            window_size_test=window_size_test,
            keys_traj=keys_traj_cfg,  # Use keys_traj_cfg directly
            keys_global=keys_global_cfg,
            pad_before=pad_before,
            pad_after=pad_after,
            pad_type=pad_type,
            random_seed=random_seed,
        )

        print(f"Train data   : {len(self.train_data)} sequences")
        print(f"Validation data: {len(self.val_data)} sequences")
        print(f"Test data    : {len(self.test_data)} sequences")