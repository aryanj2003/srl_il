import torch
from implement_dataset_base import CustomTrajectoryDataset  # Import your custom dataset class

# Define the path to your dataset and the keys for trajectory data
data_path = '/home/grimmlins/IL_workspace/data_collection_output'
keys_traj = [
    'gripper_pressure',
    'servo_node_delta_twist_cmds',
    'gripper_distance',
    'joint_states',
]


# Initialize the dataset
dataset = CustomTrajectoryDataset(data_path=data_path, keys_traj=keys_traj)

# Test the __getitem__ method by fetching a few trajectories
for idx in range(5):  # Fetch first 5 samples
    traj_dict, global_dict = dataset[idx]
    
    # Print the shapes of the data (assuming the data is in tensor form)
    print(f"Sample {idx}:")
    for key, data in traj_dict.items():
        if isinstance(data, torch.Tensor):
            print(f"  {key}: {data.shape}")
        else:
            print(f"  {key}: {len(data)}")  # If it's not a tensor, print the length (for lists)

    # Print the length of the trajectory
    print(f"  Sequence length: {dataset.get_seq_length(idx)}")

    # Check if the dataset is loaded correctly
    print(f"  Loaded successfully: {len(traj_dict) > 0}")
    print("=" * 40)

# Check the total number of trajectories
print(f"Total number of trajectories in the dataset: {len(dataset)}")
