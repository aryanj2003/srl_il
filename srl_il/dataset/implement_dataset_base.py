# implement_dataset_base.py
import time
import os, torch, logging
import numpy as np
import pandas as pd
from srl_il.dataset.dataset_base import TrajectoryDataset

# Configure logging (unchanged)
logging.basicConfig(
    filename="/home/aryan/IL_Workspace/srl_il/output/trajectory_dataset.log",
    level=logging.DEBUG,
    format="%(asctime)s [DEBUG] %(message)s"
)

class CustomTrajectoryDataset(TrajectoryDataset):
    def __init__(
        self,
        data_path: str,
        keys_traj: list,                  
        data_type: str = "train",
        joint_states_stride: int = 1,     
        preloaded: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.data_path = data_path
        self.keys_traj = keys_traj
        self.data_type = data_type
        self.joint_states_stride = joint_states_stride
        self.preloaded = preloaded

        # where files live & cached dataframes/column selections
        self._paths = {}
        self._cache = {}      # key -> DataFrame
        self._colmap = {}     # key -> ordered list of numeric columns used for that key

        # Map logical keys -> expected 20 Hz CSV filenames (fall back to <key>.csv)
        name_map = {
            "gripper_pressure": "kept_pressure_windows_20hz.csv",
            "joint_states_fixed": "kept_joint_states_windows_20hz.csv",
            "force_torque_sensor_broadcaster_wrench": "kept_force_torque_sensor_broadcaster_wrench_windows_20hz.csv",
            "servo_node_delta_twist_cmds": "kept_servo_node_delta_twist_cmds_windows_20hz.csv",
        }

        for key in keys_traj:
            print(f"Key: {key}")
            fn = name_map.get(key, f"{key}.csv")
            path = os.path.join(self.data_path, fn)
            self._paths[key] = path
            self._cache[key] = None
            self._colmap[key] = None

            if self.preloaded:
                # eagerly load
                self._cache[key] = self._load_csv(path, key)


    def _edge_fill_20hz_boundaries(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """
        Fill only leading/trailing all-NaN ticks per trial_id by copying the nearest
        real sample within ≤ dt/2. No interior fill, no extrapolation.
        Adds 'imputed_edge' (0/1). Returns a new DataFrame.
        """
        # Early outs
        if df is None or df.empty or "t_sec" not in df.columns or not cols:
            return df.copy() if df is not None else pd.DataFrame()

        # Build grouping iterator safely (avoid empty groupby -> [])
        if "trial_id" in df.columns and df["trial_id"].notna().any():
            groups_iter = df.groupby("trial_id", sort=False)
        else:
            groups_iter = [(None, df.copy())]

        out = []
        for tid, g in groups_iter:
            if isinstance(g, tuple):  
                g = g[1]
            if g.empty:
                out.append(g)
                continue

            g = g.sort_values("t_sec").reset_index(drop=True)

            # estimate grid step (fallback to 20 Hz)
            ts = g["t_sec"].to_numpy()
            if len(ts) >= 2:
                diffs = np.diff(np.unique(ts))
                dt = float(np.median(diffs)) if len(diffs) else 1.0/20.0
            else:
                dt = 1.0/20.0
            tol = dt / 2.0 + 1e-9

            allnan = g[cols].isna().all(axis=1).to_numpy()
            if (~allnan).sum() == 0:
                out.append(g); continue

            good_idx = np.where(~allnan)[0]
            first_good, last_good = good_idx[0], good_idx[-1]
            lead_mask = np.zeros(len(g), dtype=bool);  lead_mask[:first_good] = allnan[:first_good]
            trail_mask = np.zeros(len(g), dtype=bool); trail_mask[last_good+1:] = allnan[last_good+1:]
            edge_mask = lead_mask | trail_mask
            if not edge_mask.any():
                out.append(g); continue

            # ensure unique, sorted index for nearest-reindex
            src = (
                g.loc[~allnan, ["t_sec"] + cols]
                .drop_duplicates(subset=["t_sec"], keep="last")
                .set_index("t_sec")
                .sort_index()
            )
            nearest = src.reindex(g["t_sec"], method="nearest", tolerance=tol)

            idx = g.index[edge_mask]
            for c in cols:
                take = edge_mask & g[c].isna().to_numpy() & nearest[c].notna().to_numpy()
                if take.any():
                    g.loc[g.index[take], c] = nearest.loc[nearest.index[take], c].to_numpy()

            if "imputed_edge" not in g.columns:
                g["imputed_edge"] = 0
            g.loc[idx, "imputed_edge"] = 1
            g["imputed_edge"] = g["imputed_edge"].fillna(0).astype(int)

            out.append(g)

        # If nothing was appended, return an empty frame with the same columns
        if not out:
            return df.iloc[0:0].copy()

        return pd.concat(out, ignore_index=True)




    # --- helper to sanitize the values ---
    def _sanitize_numeric_by_key(self, df: pd.DataFrame, key: str) -> pd.DataFrame:
        """
        Coerce to numeric, remove inf/NaN, clip physically impossible magnitudes per stream,
        then time-aware interpolate (index = t_sec). Returns a new DataFrame with same columns.
        """
        if df.empty or "t_sec" not in df.columns:
            return df

        df = df.copy()
        # ensure numeric
        num_cols = [c for c in df.select_dtypes(include="number").columns if c != "t_sec"]
        # if pandas inferred some as object, coerce
        for c in df.columns:
            if c != "t_sec" and (c not in num_cols):
                try:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                    if c not in num_cols and df[c].dtype.kind in ("i", "u", "f"):
                        num_cols.append(c)
                except Exception:
                    pass

        # replace inf/-inf with NaN
        if num_cols:
            df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)

        # Respect per-trial order if present
        if "trial_id" in df.columns:
            df = df.sort_values(["trial_id", "t_sec"]).reset_index(drop=True)
        else:
            df = df.sort_values("t_sec").reset_index(drop=True)

        # === If this is a 20 Hz artifact CSV, do not re-interpolate; just edge-fill ===
        if "imputed" in df.columns:
            # use your chosen columns order for this key (already set in _load_csv)
            cols_for_key = [c for c in (self._colmap.get(key) or []) if c in df.columns]
            return self._edge_fill_20hz_boundaries(df, cols_for_key)
        return df


    # ---------- CSV LOADER (vectorized) ----------
    def _load_csv(self, path: str, key: str) -> pd.DataFrame:
        print(f"Loading {path} …")
        if not os.path.exists(path):
            logging.error(f"CSV not found for key={key}: {path}")
            return pd.DataFrame()
        df = pd.read_csv(path)
        if "t_sec" not in df.columns:
            logging.error(f"Missing t_sec in CSV for key={key}: {path}")
            return pd.DataFrame()

        # --- Pick the exact numeric columns (order matters for your per-key vector) ---
        cols = []
        if key == "gripper_pressure":
            candidates = [c for c in df.columns if c in ("data_0", "data_1", "data_2")]
            cols = sorted(candidates, key=lambda x: ["data_0","data_1","data_2"].index(x))  
            if not cols:
                cols = [c for c in df.columns if c.startswith("pressure_")]
        elif key == "joint_states_fixed":
            # Known 6-DoF order; adjust if your robot differs
            JOINT_RANK = {
                "shoulder_pan_joint": 0,
                "shoulder_lift_joint": 1,
                "elbow_joint": 2,
                "wrist_1_joint": 3,
                "wrist_2_joint": 4,
                "wrist_3_joint": 5,
            }

            pos_cols = [c for c in df.columns if c.startswith("position_")]
            vel_cols = [c for c in df.columns if c.startswith("velocity_")]
            eff_cols = [c for c in df.columns if c.startswith("effort_")]

            # Sort rule: if the last token is a number → sort by that number.
            # Otherwise, strip the prefix and sort by JOINT_RANK; unknown names go after known ones.
            pos = sorted(
                pos_cols,
                key=lambda s: (
                    0, int(s.split("_")[-1])
                ) if s.split("_")[-1].isdigit() else (
                    1, JOINT_RANK.get(s.split("position_")[-1], 999), s
                )
            )
            vel = sorted(
                vel_cols,
                key=lambda s: (
                    0, int(s.split("_")[-1])
                ) if s.split("_")[-1].isdigit() else (
                    1, JOINT_RANK.get(s.split("velocity_")[-1], 999), s
                )
            )
            eff = sorted(
                eff_cols,
                key=lambda s: (
                    0, int(s.split("_")[-1])
                ) if s.split("_")[-1].isdigit() else (
                    1, JOINT_RANK.get(s.split("effort_")[-1], 999), s
                )
            )
            pos = pos[:12] + [f"__pad_pos_{i}" for i in range(max(0, 12 - len(pos)))]
            vel = vel[:6]  + [f"__pad_vel_{i}" for i in range(max(0, 6  - len(vel)))]
            eff = eff[:6]  + [f"__pad_eff_{i}" for i in range(max(0, 6  - len(eff)))]
            cols = pos + vel + eff  
            cols = pos + vel + eff
        elif key == "force_torque_sensor_broadcaster_wrench":
            preferred = ["force_x","force_y","force_z","torque_x","torque_y","torque_z"]
            cols = [c for c in preferred if c in df.columns]
            if len(cols) < 6:
                numeric = [c for c in df.select_dtypes(include="number").columns if c != "t_sec"]
                cols = numeric[:6]
        elif key == "servo_node_delta_twist_cmds":
            preferred = ["linear_x","linear_y","linear_z","angular_x","angular_y","angular_z"]
            cols = [c for c in preferred if c in df.columns]
            if len(cols) < 6:
                numeric = [c for c in df.select_dtypes(include="number").columns if c != "t_sec"]
                cols = numeric[:6]
        else:
            cols = [c for c in df.select_dtypes(include="number").columns if c not in ("t_sec","imputed")]
        self._colmap[key] = cols


        # Sort by trial then time if available
        if "trial_id" in df.columns:
            df = (
        df.sort_values(["trial_id", "t_sec"], kind="mergesort")
          .drop_duplicates(subset=["trial_id", "t_sec"], keep="last")
          .reset_index(drop=True)
    )
        else:
            df = df.sort_values("t_sec", kind="mergesort").reset_index(drop=True)

        df = self._sanitize_numeric_by_key(df, key)

        return df

    def __len__(self):
        # length of your first trajectory stream
        first = self.keys_traj[0]
        if self._cache[first] is None:
            self._cache[first] = self._load_csv(self._paths[first], first)
        return len(self._cache[first])

    def __getitem__(self, idx):
        logging.debug(f"CustomTrajectoryDataset.__getitem__: idx={idx}")
        print(f"[DEBUG] CustomTrajectoryDataset.__getitem__: idx={idx}")

        traj_dict = {}
        global_dict = {}

        # Dynamically compute expected dims (from _colmap) for robust checks
        def _expected_cols_for(key: str) -> int:
            cols = self._colmap.get(key) or []
            return len(cols)

        for key in self.keys_traj:
            if self._cache[key] is None:
                self._cache[key] = self._load_csv(self._paths[key], key)
            df = self._cache[key]

            if idx >= len(df):
                logging.error(f"Index out of range for {key} at idx={idx}, len(cache)={len(df)}")
                print(f"[ERROR] Index out of range for {key} at idx={idx}, len(cache)={len(df)}")
                raise IndexError(f"Index {idx} out of range for {key}")

            # pull row vector (list of floats) in the fixed column order selected at load time
            cols = self._colmap.get(key) or []
            rec = df.iloc[idx]

            logging.debug(f"Processing key={key}, idx={idx}, rec(t_sec)={rec.get('t_sec', 'NA')}")
            print(f"[DEBUG] Processing key={key}, idx={idx}, rec(t_sec)={rec.get('t_sec', 'NA')}")

            try:
                if not cols:
                    data = []
                else:
                    vals = []
                    for c in cols:
                        if c.startswith("__pad_"):     
                            vals.append(0.0)
                        else:
                            v = rec.get(c, np.nan)
                            vals.append(float(v) if pd.notna(v) else float('nan'))

                    data = vals

                exp = _expected_cols_for(key)
                if len(data) != exp:
                    logging.warning(f"Invalid {key} length at idx={idx}: got {len(data)}, expected {exp}; padding/truncating.")
                    print(f"[WARNING] Invalid {key} length at idx={idx}: got {len(data)}, expected {exp}; padding/truncating.")
                    # pad/truncate to expected length
                    if len(data) < exp:
                        data = data + [0.0] * (exp - len(data))
                    else:
                        data = data[:exp]
            except Exception as e:
                logging.error(f"Error processing {key} at idx={idx}: {e}.")
                print(f"[ERROR] Error processing {key} at idx={idx}: {e}.")
                data = [0.0] * _expected_cols_for(key)

            logging.debug(f"Before load: key={key}, data={data}, type={type(data)}, shape={np.array(data).shape if isinstance(data, (list, np.ndarray)) else 'N/A'}")
            print(f"[DEBUG] Before load: key={key}, data={data}, type={type(data)}, shape={np.array(data).shape if isinstance(data, (list, np.ndarray)) else 'N/A'}")

            traj_dict[key] = data

        logging.debug(f"__getitem__ end: idx={idx}, traj_dict.keys={traj_dict.keys()}, shapes={[np.array(v).shape for k, v in traj_dict.items()]}")
        print(f"[DEBUG] CustomTrajectoryDataset.__getitem__: idx={idx}, traj_dict.keys={traj_dict.keys()}")
        for key, value in traj_dict.items():
            print(f"[DEBUG] Key={key}, type={type(value)}, shape={np.array(value).shape if isinstance(value, (list, np.ndarray)) else 'N/A'}")

        return traj_dict, global_dict

    def get_seq_length(self, idx):
        # unchanged
        return len(self)

    def load(self, data, key, is_global=False):
        logging.debug(
            f"DatasetBase.load: key={key}, is_global={is_global}, "
            f"data_type={type(data)}, sample={data[:2] if isinstance(data, list) else (data.shape if hasattr(data,'shape') else type(data))}"
        )
        if isinstance(data, torch.Tensor):
            return data.to(dtype=torch.float32)
        if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
            t = torch.tensor(data, dtype=torch.float32)
            return t.unsqueeze(-1)
        if isinstance(data, list) and all(isinstance(x, list) for x in data):
            return torch.tensor(data, dtype=torch.float32)
        return torch.as_tensor(data, dtype=torch.float32)
