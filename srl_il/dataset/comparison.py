import pandas as pd
import numpy as np
from pathlib import Path

base = Path("/mnt/data")  # adjust if needed
paths = {
    "gripper_pressure": base / "/home/aryan/Documents/pressure_servo_20250725_115412.db3_export/_gripper_pressure.csv",
    "joint_states_fixed": base / "/home/aryan/Documents/pressure_servo_20250725_115412.db3_export/_joint_states.csv",
    "force_torque_sensor_broadcaster_wrench": base / "/home/aryan/Documents/pressure_servo_20250725_115412.db3_export/_force_torque_sensor_broadcaster_wrench.csv",
    "servo_node_delta_twist_cmds": base / "/home/aryan/Documents/pressure_servo_20250725_115412.db3_export/_servo_node_delta_twist_cmds.csv",
}

def load_csv(path):
    df = pd.read_csv(path)  # uses header row from your files
    time = df["t_sec"].values if "t_sec" in df.columns else None
    data = df.drop(columns=["t_sec"]) if "t_sec" in df.columns else df
    # force numeric (bad cells -> NaN)
    data = data.apply(pd.to_numeric, errors="coerce")
    return time, data

def nearest_row_by_time(time, data, t_query):
    if time is None: 
        return None, None
    i = int(np.argmin(np.abs(time - t_query)))
    return i, data.iloc[i].to_list()

streams = {name: load_csv(p) for name, p in paths.items()}

# Example: compare to times from your logs
for label, t_query in {10:3.736322156, 152:8.886322156, 216:10.486322156, 241:11.109096119, 286:12.236322156}.items():
    print(f"\n=== t≈{t_query} (from idx {label}) ===")
    for name, (t, data) in streams.items():
        i, row = nearest_row_by_time(t, data, t_query)
        if i is None:
            print(f"{name:35s} -> no t_sec column")
        else:
            print(f"{name:35s} -> row {i}, values: {row}")
