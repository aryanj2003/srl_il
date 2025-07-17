import yaml
import torch
from srl_il.dataset.dataset_base import TrajectoryDataset
import os
import yaml
import array

def array_constructor(loader, node):
    # node.value comes in as [ typecode, [values...] ]
    typecode, values = loader.construct_sequence(node)
    return array.array(typecode, values)

# Register on the SafeLoader before any yaml.load calls
yaml.SafeLoader.add_constructor(
    'tag:yaml.org,2002:python/object/apply:array.array',
    array_constructor
)



class CustomTrajectoryDataset(TrajectoryDataset):
    def __init__(self, data_path, keys_traj, data_type="train", **kwargs):
        self.data_path = data_path
        self.keys_traj = keys_traj
        self.data_type = data_type
        self.dataset = {}

        for key in keys_traj:
            filename = f"_{key}.yaml"                
            path = os.path.join(self.data_path, filename)
            print(f"YAML file path: {path}")
            content = open(path, 'r').read()
            if not content:
                print(f"Warning: Empty file {path}")
            print(f"Content of {path}: {content[:100]}")

            try:
                loaded_data = yaml.load(content, Loader=yaml.SafeLoader)
                print(f"Loaded {len(loaded_data)} records for {key}")

                # convert any list-values inside each record to tuples
                for rec in loaded_data:
                    if isinstance(rec, dict):
                        for k,v in rec.items():
                            if isinstance(v, list):
                                rec[k] = tuple(v)

                self.dataset[key] = loaded_data

            except Exception as e:
                print(f"Error reading {path}: {e}")

    def __getitem__(self, idx):
        traj_dict, global_dict = {}, {}
        for key in self.keys_traj:
            records = self.dataset.get(key, [])
            if idx >= len(records):
                print(f"Index {idx} out of range for {key}")
                continue
            rec = records[idx]

            if key == "servo_node_delta_twist_cmds":
                traj_dict[key] = rec['action']['twist_cmd']
            elif key in ["gripper_distance", "gripper_pressure"]:
                traj_dict[key] = rec['obs']
        return traj_dict, global_dict

    def get_seq_length(self, idx):
        return len(self.dataset[self.keys_traj[0]])


    def __len__(self):
        return len(self.dataset[self.keys_traj[0]])

    def load(self, data, key, is_global=False):
        return torch.tensor(data, dtype=torch.float32)
