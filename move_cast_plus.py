# ============================================================
# MoveCast+ (Evaluation-API Ready) — GRU+GBM (offline) + Kinematic Fallback (online)
# ============================================================
# - Offline (DO_TRAIN=True): trains GRU+LightGBM on historic data.
# - Online (Evaluation API): predicts without leakage, using kinematic fallback by default.
#   Later, set USE_TRAINED_MODELS=True and provide your trained artifacts to blend GRU+GBM.
# ============================================================

from __future__ import annotations

import os
import glob
import time
import math
import random
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import joblib
import pickle
import lightgbm as lgb
import numpy as np
import pandas as pd
try:
    import polars as pl
except ModuleNotFoundError:  # pragma: no cover - polars is available on Kaggle but guard for local runs
    pl = None
import torch
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")

# -------------------------
# Submission / Training Flags
# -------------------------
KAGGLE_SUBMIT = False              # running inside Kaggle submission
DO_TRAIN = True                  # set True only in a training notebook/session
USE_TRAINED_MODELS = True       # set True if you uploaded trained GRU/GBM to input paths below

# Trained model artifact paths (when USE_TRAINED_MODELS=True)
# When using pre-trained models from /kaggle/input/, save new artifacts in /kaggle/working/
LGBM_DIR = "/kaggle/working/lgbm_models"
os.makedirs(LGBM_DIR, exist_ok=True)

# ROLE_SCALERS_PATH should *always* be writable
ROLE_SCALERS_PATH = os.path.join(LGBM_DIR, "role_scalers.pkl")
MODEL_CHECKPOINT = "best_model_gru.pth"
LGBM_X_FILES = ["/kaggle/input/models/lgbm_x_seed42.pkl","/kaggle/input/models/lgbm_x_seed43.pkl","/kaggle/input/models/lgbm_x_seed44.pkl"]  # e.g., ["/kaggle/input/your-dataset/lgbm_x_seed42.pkl", ...]
LGBM_Y_FILES = ["/kaggle/input/models/lgbm_y_seed42.pkl","/kaggle/input/models/lgbm_y_seed43.pkl","/kaggle/input/models/lgbm_y_seed44.pkl"]  # e.g., ["/kaggle/input/your-dataset/lgbm_y_seed42.pkl", ...]
GRU_CHECKPOINT_FILE = ""  # e.g., "/kaggle/input/your-dataset/best_model_gru.pth"

# -------------------------
# Runtime Config / Reproducibility
# -------------------------
NUM_WORKERS = 4
os.environ.update({"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"})
torch.set_num_threads(1)

SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

# -------------------------
# Hyperparams
# -------------------------
DEBUG_MODE = False
DEBUG_SAMPLE_FRAC = 0.02

BATCH_SIZE = 192
MAX_EPOCHS = 150
MAX_LR = 1e-3
WEIGHT_DECAY = 5e-5
CLIP_VALUE = 2.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRU_HIDDEN = 512
GRU_WEIGHT, GBM_WEIGHT = 0.8, 0.2
N_NEIGHBORS = 5

GRU_LR = 1e-3 if DEBUG_MODE else MAX_LR
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    pass

print(f"Using device: {device}")
print(f"KAGGLE_SUBMIT={KAGGLE_SUBMIT}, DO_TRAIN={DO_TRAIN}, USE_TRAINED_MODELS={USE_TRAINED_MODELS}")

# ============================================================
# Helpers
# ============================================================
def format_runtime(s):
    m, sec = divmod(int(s), 60); h, m = divmod(m, 60); return f"{h}h {m}m"

def height_to_inches(x):
    try: f,i = str(x).split("-"); return int(f)*12+int(i)
    except: return 0

def safe_div(a,b,eps=1e-6): return a/(b+eps)

def deg2rad(d): return np.deg2rad(d.astype(np.float32))


def ensure_pandas(df):
    if isinstance(df, pd.DataFrame):
        return df.copy()
    if pl is not None and isinstance(df, pl.DataFrame):
        return df.to_pandas().copy()
    raise TypeError(f"Unsupported dataframe type: {type(df)!r}")

# ============================================================
# Feature / Engineering helpers (used offline; some reused online)
# ============================================================
def compute_raw_ball_cols(df, bx_col="ball_land_x", by_col="ball_land_y"):
    if bx_col in df.columns: bx = pd.to_numeric(df[bx_col], errors="coerce").fillna(0.0).to_numpy(np.float32)
    else: bx = np.zeros(len(df), np.float32)
    if by_col in df.columns: by = pd.to_numeric(df[by_col], errors="coerce").fillna(0.0).to_numpy(np.float32)
    else: by = np.zeros(len(df), np.float32)
    raw_bx = bx * 120.0 if (len(bx) and np.nanmax(bx) <= 1.1) else bx.copy()
    raw_by = by * 53.3 if (len(by) and np.nanmax(by) <= 1.1) else by.copy()
    return raw_bx, raw_by

def add_engineered_features(df):
    df = df.copy()
    dir_rad = np.deg2rad(pd.to_numeric(df.get("dir", 0), errors="coerce").fillna(0.0))
    df["heading_x"] = np.sin(dir_rad)
    df["heading_y"] = np.cos(dir_rad)
    dt = 0.1
    s = pd.to_numeric(df.get("s", 0), errors="coerce").fillna(0.0)
    a = pd.to_numeric(df.get("a", 0), errors="coerce").fillna(0.0)
    vmag = (s + 0.5 * a * dt)
    df["velocity_x"] = vmag * df["heading_x"]
    df["velocity_y"] = vmag * df["heading_y"]
    df["acceleration_x"] = a * df["heading_x"]
    df["acceleration_y"] = a * df["heading_y"]

    if "ball_land_x_raw" in df.columns and "ball_land_y_raw" in df.columns:
        dx = df["ball_land_x_raw"] - df.get("x", 0.0); dy = df["ball_land_y_raw"] - df.get("y", 0.0)
        dist = np.sqrt(np.maximum(dx*dx + dy*dy, 0.0))
        df["dist_to_ball"] = dist
        df["angle_to_ball"] = np.arctan2(dy, dx)
        df["ball_dir_x"] = dx / (dist + 1e-6)
        df["ball_dir_y"] = dy / (dist + 1e-6)
        vdotb = df["velocity_x"] * df["ball_dir_x"] + df["velocity_y"] * df["ball_dir_y"]
        df["closing_speed_to_ball"] = vdotb
        df["vx_rel_ball"] = (df["velocity_x"] - vdotb * df["ball_dir_x"]).fillna(0.0)
        df["vy_rel_ball"] = (df["velocity_y"] - vdotb * df["ball_dir_y"]).fillna(0.0)
    else:
        df["dist_to_ball"] = 0.0; df["angle_to_ball"] = 0.0
        df["ball_dir_x"] = 0.0; df["ball_dir_y"] = 0.0
        df["closing_speed_to_ball"] = 0.0
        df["vx_rel_ball"] = 0.0; df["vy_rel_ball"] = 0.0

    df["is_offense"] = (df.get("player_side","").astype(str).str.upper().str.startswith("OFF")).astype(float)
    df["is_defense"] = (df.get("player_side","").astype(str).str.upper().str.startswith("DEF")).astype(float)
    df["is_receiver"] = (df.get("player_role","").astype(str).str.contains("Receiver", na=False)).astype(float)
    df["is_passer"] = (df.get("player_role","").astype(str).str.contains("Passer", na=False)).astype(float)
    return df

# ============================================================
# Kinematic fallback predictor (safe for Evaluation API)
# ============================================================
def kinematic_predict(test_chunk: pd.DataFrame, sample_prediction_df: pd.DataFrame, dt=0.1) -> pd.DataFrame:
    """ Kinematic fallback prediction. """
    test_chunk = test_chunk.copy()
    for c in ["x","y","s","a","dir"]:
        if c in test_chunk.columns:
            test_chunk[c] = pd.to_numeric(test_chunk[c], errors="coerce").fillna(0.0)
        else:
            test_chunk[c] = 0.0

    test_chunk = test_chunk.sort_values(["game_id","play_id","nfl_id","frame_id"])
    last_state = test_chunk.groupby(["game_id","play_id","nfl_id"], as_index=False).tail(1)
    meta = sample_prediction_df[["game_id","play_id","nfl_id"]].drop_duplicates()
    state = meta.merge(
        last_state[["game_id","play_id","nfl_id","x","y","s","a","dir"]],
        on=["game_id","play_id","nfl_id"], how="left"
    ).fillna(0.0)

    out = sample_prediction_df.copy()
    out["x"] = 0.0
    out["y"] = 0.0

    state_dict = {}
    for r in state.itertuples(index=False):
        key = (int(r.game_id), int(r.play_id), int(r.nfl_id))
        hx = math.sin(math.radians(float(r.dir)))
        hy = math.cos(math.radians(float(r.dir)))
        state_dict[key] = {
            "x": float(r.x), "y": float(r.y),
            "s": float(r.s), "a": float(r.a),
            "hx": hx, "hy": hy
        }

    speed_decay = 0.98

    rows = []
    for r in out.itertuples(index=False):
        key = (int(r.game_id), int(r.play_id), int(r.nfl_id))
        f = int(r.frame_id)
        st = state_dict.get(key, {"x":0.0,"y":0.0,"s":0.0,"a":0.0,"hx":0.0,"hy":1.0}).copy()
        x, y, s, a, hx, hy = st["x"], st["y"], st["s"], st["a"], st["hx"], st["hy"]
        for _ in range(f):
            v = s + 0.5 * a * dt
            x += v * hx * dt
            y += v * hy * dt
            s = s * speed_decay + a * dt
            a = a * 0.98
        rows.append([r.game_id, r.play_id, r.nfl_id, r.frame_id, x, y])

    pred = pd.DataFrame(rows, columns=["game_id","play_id","nfl_id","frame_id","x","y"])
    return pred

# ============================================================
# (Offline) Full training pipeline (kept for you to develop accuracy)
# ============================================================
def offline_training_pipeline():
    input_pattern  = "/kaggle/input/nfl-big-data-bowl-2026-prediction/train/input_2023_w*.csv"
    output_pattern = "/kaggle/input/nfl-big-data-bowl-2026-prediction/train/output_2023_w*.csv"
    test_input_path_local = "/kaggle/input/nfl-big-data-bowl-2026-prediction/test_input.csv"
    test_csv_path_local   = "/kaggle/input/nfl-big-data-bowl-2026-prediction/test.csv"

    start_total = time.time()
    start_data  = time.time()

    input_files = sorted(glob.glob(input_pattern))
    if not input_files: raise FileNotFoundError(f"No input files matched: {input_pattern}")
    output_files = sorted(glob.glob(output_pattern))
    if not output_files: raise FileNotFoundError(f"No output files matched: {output_pattern}")

    input_df  = pd.concat([pd.read_csv(f, low_memory=False) for f in input_files], ignore_index=True)
    output_df = pd.concat([pd.read_csv(f, low_memory=False) for f in output_files], ignore_index=True)

    test_input_df = pd.read_csv(test_input_path_local, low_memory=False)
    test_csv_df   = pd.read_csv(test_csv_path_local, low_memory=False)

    rename_map = {"player_position": "position", "frameId": "frame_id"}
    test_input_df.rename(columns={k:v for k,v in rename_map.items() if k in test_input_df.columns}, inplace=True)
    test_csv_df.rename(columns={k:v for k,v in rename_map.items() if k in test_csv_df.columns}, inplace=True)

    if DEBUG_MODE:
        game_ids = input_df['game_id'].unique()
        sample_games = np.random.choice(game_ids, size=max(1, int(len(game_ids)*DEBUG_SAMPLE_FRAC)), replace=False)
        input_df  = input_df[input_df['game_id'].isin(sample_games)].reset_index(drop=True)
        output_df = output_df[output_df['game_id'].isin(sample_games)].reset_index(drop=True)

    merged_df = pd.merge(
        input_df, output_df,
        on=["game_id","play_id","nfl_id","frame_id"],
        how="left", suffixes=("", "_target")
    )

    end_data = time.time()
    print(f"Train rows: {len(merged_df)} | Test rows: {len(test_input_df)}")
    print(f"✅ Normalized test_input_df columns: {sorted([c for c in test_input_df.columns if 'frame' in c or 'pos' in c or 'player_' in c])}")
    print(f"Data Loading & Merging completed in {format_runtime(end_data - start_data)}")

    merged_df['x_play_mean'] = merged_df.groupby(['game_id','play_id'])['x'].transform('mean')
    merged_df['y_play_mean'] = merged_df.groupby(['game_id','play_id'])['y'].transform('mean')
    merged_df['x_centered'] = merged_df['x'] - merged_df['x_play_mean']
    merged_df['y_centered'] = merged_df['y'] - merged_df['y_play_mean']

    if {'game_id','play_id','x'}.issubset(test_input_df.columns):
        test_input_df['x_play_mean'] = test_input_df.groupby(['game_id','play_id'])['x'].transform('mean')
        test_input_df['y_play_mean'] = test_input_df.groupby(['game_id','play_id'])['y'].transform('mean')
        test_input_df['x_centered'] = test_input_df['x'] - test_input_df['x_play_mean']
        test_input_df['y_centered'] = test_input_df['y'] - test_input_df['y_play_mean']
    else:
        test_input_df['x_centered'] = test_input_df.get('x', 0.0)
        test_input_df['y_centered'] = test_input_df.get('y', 0.0)

    print(f"Train merged rows: {len(merged_df)}, Test input rows: {len(test_input_df)}")

    merged_df["player_height_inches"] = merged_df.get("player_height", "0-0").apply(height_to_inches)
    merged_df["player_weight"] = pd.to_numeric(merged_df.get("player_weight", 0), errors="coerce").fillna(0)

    if "absolute_yardline_number" in merged_df.columns and "play_direction" in merged_df.columns:
        merged_df.loc[merged_df["play_direction"] == 0, "absolute_yardline_number"] = \
            120 - merged_df.loc[merged_df["play_direction"] == 0, "absolute_yardline_number"]

    merged_df["play_direction"] = merged_df.get("play_direction","right").astype(str).str.lower().map({"right":1,"left":0}).fillna(1).astype(np.float32)
    merged_df["player_side"] = merged_df.get("player_side","UNK").astype(str)
    merged_df["player_role"] = merged_df.get("player_role","UNK").astype(str)
    merged_df["position"] = merged_df.get("player_position", merged_df.get("position","UNK")).astype(str)

    if "play_direction" in test_input_df.columns:
        test_input_df["play_direction"] = test_input_df["play_direction"].astype(str).str.lower().map({"right":1,"left":0}).fillna(1).astype(np.float32)
    else:
        test_input_df["play_direction"] = 1.0
    if "player_side" not in test_input_df.columns: test_input_df["player_side"] = "UNK"
    if "player_role" not in test_input_df.columns: test_input_df["player_role"] = "UNK"
    if "position" not in test_input_df.columns:   test_input_df["position"] = test_input_df.get("player_position", "UNK")

    numeric_cols = ["x","y","s","a","dir","o","player_height_inches","player_weight",
                    "absolute_yardline_number","ball_land_x","ball_land_y"]
    for col in numeric_cols:
        if col in merged_df.columns: merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").fillna(0.0)
        if col in test_input_df.columns: test_input_df[col] = pd.to_numeric(test_input_df[col], errors="coerce").fillna(0.0)

    merged_bx_raw, merged_by_raw = compute_raw_ball_cols(merged_df)
    merged_df["ball_land_x_raw"] = merged_bx_raw
    merged_df["ball_land_y_raw"] = merged_by_raw
    test_bx_raw, test_by_raw = compute_raw_ball_cols(test_input_df)
    test_input_df["ball_land_x_raw"] = test_bx_raw
    test_input_df["ball_land_y_raw"] = test_by_raw

    merged_df["x_rel_ball"] = merged_df["x"] - merged_df["ball_land_x_raw"]
    merged_df["y_rel_ball"] = merged_df["y"] - merged_df["ball_land_y_raw"]
    if "x" in test_input_df.columns:
        test_input_df["x_rel_ball"] = test_input_df["x"] - test_input_df["ball_land_x_raw"]
        test_input_df["y_rel_ball"] = test_input_df["y"] - test_input_df["ball_land_y_raw"]
    else:
        test_input_df["x_rel_ball"] = 0.0; test_input_df["y_rel_ball"] = 0.0

    merged_df = add_engineered_features(merged_df)
    test_input_df = add_engineered_features(test_input_df)

    position_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    position_encoder.fit(merged_df[["position"]])
    pos_ohe_arr = position_encoder.transform(merged_df[["position"]]).astype(np.float32)

    side_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    side_encoder.fit(merged_df[["player_side"]])
    side_ohe_arr = side_encoder.transform(merged_df[["player_side"]]).astype(np.float32)

    role_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    role_encoder.fit(merged_df[["player_role"]])
    role_ohe_arr = role_encoder.transform(merged_df[["player_role"]]).astype(np.float32)

    pos_ohe_test  = position_encoder.transform(test_input_df[["position"]]).astype(np.float32)
    side_ohe_test = side_encoder.transform(test_input_df[["player_side"]]).astype(np.float32)
    role_ohe_test = role_encoder.transform(test_input_df[["player_role"]]).astype(np.float32)

    if "absolute_yardline_number" in merged_df.columns: merged_df["absolute_yardline_number"] = merged_df["absolute_yardline_number"] / 120.0
    if "ball_land_x" in merged_df.columns: merged_df["ball_land_x"] = merged_df["ball_land_x"] / 120.0
    if "ball_land_y" in merged_df.columns: merged_df["ball_land_y"] = merged_df["ball_land_y"] / 53.3
    if "absolute_yardline_number" in test_input_df.columns: test_input_df["absolute_yardline_number"] = test_input_df["absolute_yardline_number"] / 120.0
    if "ball_land_x" in test_input_df.columns: test_input_df["ball_land_x"] = test_input_df["ball_land_x"] / 120.0
    if "ball_land_y" in test_input_df.columns: test_input_df["ball_land_y"] = test_input_df["ball_land_y"] / 53.3

    base_cols = [
        "x_centered","y_centered","s","a","dir","o",
        "player_height_inches","player_weight",
        "absolute_yardline_number","ball_land_x","ball_land_y",
        "x_rel_ball","y_rel_ball",
        "heading_x","heading_y","velocity_x","velocity_y",
        "acceleration_x","acceleration_y",
        "dist_to_ball","angle_to_ball","closing_speed_to_ball",
        "is_offense","is_defense","is_receiver","is_passer"
    ]
    for c in base_cols:
        if c not in merged_df.columns: merged_df[c] = 0.0
        if c not in test_input_df.columns: test_input_df[c] = 0.0

    role_scalers = {}
    numeric_for_role_scaling = base_cols.copy()
    global_vals = merged_df[numeric_for_role_scaling].to_numpy(dtype=np.float32)
    global_role_scaler = RobustScaler()
    if len(global_vals) > 1:
        global_role_scaler.fit(global_vals)
    else:
        global_role_scaler.center_ = np.zeros(len(numeric_for_role_scaling), dtype=np.float32)
        global_role_scaler.scale_ = np.ones(len(numeric_for_role_scaling), dtype=np.float32)

    unique_roles = merged_df["player_role"].fillna("UNK").unique().tolist()
    for role in unique_roles:
        sel = merged_df["player_role"] == role
        rows = merged_df.loc[sel, numeric_for_role_scaling].to_numpy(dtype=np.float32)
        if rows.shape[0] >= 5:
            scaler_r = StandardScaler(); scaler_r.fit(rows)
            role_scalers[role] = scaler_r
        else:
            role_scalers[role] = global_role_scaler

    with open(ROLE_SCALERS_PATH, "wb") as f:
        pickle.dump(role_scalers, f)

    from joblib import Parallel, delayed
    def build_sequences(df, pos_ohe_all, side_ohe_all, role_ohe_all, n_neighbors=N_NEIGHBORS, debug_limit=None):
        df_local = df.copy().reset_index(drop=False)
        base_arr = df_local[base_cols].to_numpy(np.float32)
        roles_arr = df_local.get("player_role","UNK").fillna("UNK").to_numpy()
        for role, scaler_r in role_scalers.items():
            idxs = (roles_arr == role).nonzero()[0]
            if idxs.size > 0:
                try: base_arr[idxs] = scaler_r.transform(base_arr[idxs])
                except Exception: base_arr[idxs] = global_role_scaler.transform(base_arr[idxs])

        index_arr = df_local["index"].to_numpy()
        pos_ohe_local  = pos_ohe_all[index_arr]
        side_ohe_local = side_ohe_all[index_arr]
        role_ohe_local = role_ohe_all[index_arr]

        plays = list(df_local.groupby(["game_id","play_id"], sort=False))
        if debug_limit: plays = plays[:debug_limit]

        def process_player(frames):
            frames = frames.sort_values("frame_id").reset_index(drop=True)
            T = len(frames)
            if T == 0: return None
            init_x = float(frames.iloc[0]["x"]); init_y = float(frames.iloc[0]["y"])
            if float(frames.iloc[0].get("play_direction",1.0)) == 0.0:
                init_x = 120.0 - init_x
            x_arr = frames["x"].to_numpy(np.float32); y_arr = frames["y"].to_numpy(np.float32)
            y_seq = np.stack([x_arr - init_x, y_arr - init_y], axis=1).astype(np.float32)
            y_seq[:, 0] /= 120.0
            y_seq[:, 1] /= 53.3

            gidx = frames["index"].to_numpy()
            F_base = base_arr[gidx]
            F_pos  = pos_ohe_local[gidx]
            F_side = side_ohe_local[gidx]
            F_role = role_ohe_local[gidx]

            def rolling_mean_std(x):
                if x.size == 0: return np.zeros((0,2), np.float32)
                cs = np.cumsum(x, dtype=np.float64); cs2 = np.cumsum((x.astype(np.float64))**2)
                idx = np.arange(1, x.size+1)
                mean = cs / idx
                var = np.maximum(cs2/idx - mean**2, 0.0)
                std = np.sqrt(var, dtype=np.float64)
                return np.stack([mean, std], axis=1).astype(np.float32)

            s = frames["s"].to_numpy(np.float32)
            a = frames["a"].to_numpy(np.float32)
            d = frames["dir"].to_numpy(np.float32)
            s_ms = rolling_mean_std(s)
            a_ms = rolling_mean_std(a)
            d_std = rolling_mean_std(d)[:,1:2]
            roll_feats = np.concatenate([s_ms, a_ms, d_std], axis=1)

            dx = np.diff(x_arr, prepend=x_arr[0]); dy = np.diff(y_arr, prepend=y_arr[0])
            seg_len = np.sqrt(dx**2 + dy**2)
            path_len = np.cumsum(seg_len)
            disp = np.sqrt((x_arr - x_arr[0])**2 + (y_arr - y_arr[0])**2); disp = np.maximum(disp, 1e-6)
            sinuosity = (path_len / disp).astype(np.float32)
            jerk = np.zeros_like(a, dtype=np.float32)
            if a.size >= 3:
                sw = np.lib.stride_tricks.sliding_window_view(a, 3)
                j = np.std(sw, axis=-1).astype(np.float32)
                jerk = np.pad(j, (2,0))
            traj_feats = np.stack([sinuosity, jerk], axis=1)

            sep_feats = np.zeros((T,1), np.float32)

            def kshift(arr, k):
                z = np.zeros_like(arr)
                if k > 0: z[k:] = arr[:-k]
                else: z[:] = arr
                return z
            delta_feats = []
            for kk in (1,2,3,4,5,6):
                px = kshift(x_arr, kk); py = kshift(y_arr, kk)
                dxp = (x_arr - px); dyp = (y_arr - py)
                delta_feats.append(np.stack([dxp, dyp, dxp, dyp], axis=1))
            delta_feats = np.concatenate(delta_feats, axis=1).astype(np.float32)

            if "vx_rel_ball" in frames.columns and "vy_rel_ball" in frames.columns:
                relvel_mat = frames[["vx_rel_ball","vy_rel_ball"]].to_numpy(np.float32)
            else:
                relvel_mat = np.zeros((T,2), dtype=np.float32)

            feat = np.concatenate([
                F_base, F_pos, F_side, F_role,
                roll_feats, traj_feats, sep_feats, delta_feats, relvel_mat
            ], axis=1).astype(np.float32)

            return feat, y_seq, {"game_id": int(frames.iloc[0]["game_id"]),
                                "play_id": int(frames.iloc[0]["play_id"]),
                                "nfl_id": int(frames.iloc[0]["nfl_id"]),
                                "player_role": str(frames.iloc[0]["player_role"])}

        def process_play(play_key_df):
            (_, _), play_df = play_key_df
            out = []
            for _, frames in play_df.groupby("nfl_id", sort=False):
                res = process_player(frames)
                if res is not None:
                    out.append(res)
            return out

        parallel = Parallel(n_jobs=NUM_WORKERS, backend="loky", prefer="threads")
        play_batches = parallel(delayed(process_play)(pl) for pl in plays)

        X_list, y_list, meta = [], [], []
        for out in play_batches:
            for feat, y_seq, m in out:
                X_list.append(feat); y_list.append(y_seq); meta.append(m)
        return X_list, y_list, meta

    print("Building sequences (this can take a while)...")
    X, y, meta = build_sequences(merged_df, pos_ohe_arr, side_ohe_arr, role_ohe_arr, n_neighbors=N_NEIGHBORS)
    print(f"Built {len(X)} sequences.")
    print(f"Feature Extraction time: {format_runtime(time.time() - end_data)}")

    if len(X) < 2:
        print("⚠️  Not enough sequences were built to perform a train/validation split. Skipping offline training.")
        return

    for i in range(len(X)):
        X[i] = np.nan_to_num(X[i], nan=0.0, posinf=1e3, neginf=-1e3)
    for i in range(len(y)):
        y[i] = np.nan_to_num(y[i], nan=0.0, posinf=1e3, neginf=-1e3)

    X_train, X_val, y_train, y_val, meta_train, meta_val = train_test_split(
        X, y, meta, test_size=0.2, random_state=SEED
    )

    def flatten_sequences_for_gbm(X_seqs, y_seqs, meta_list, max_rows=None):
        rows, yx, yy = [], [], []
        count = 0
        for seq, y_seq, m in zip(X_seqs, y_seqs, meta_list):
            T = seq.shape[0]
            for t in range(T):
                rows.append(seq[t])
                yx.append(y_seq[t, 0]); yy.append(y_seq[t, 1])
                count += 1
                if max_rows and count >= max_rows:
                    return np.vstack(rows), np.array(yx, np.float32), np.array(yy, np.float32)
        if not rows:
            feature_dim = X_seqs[0].shape[1] if X_seqs else 0
            empty_features = np.zeros((0, feature_dim), dtype=np.float32)
            return empty_features, np.array([], np.float32), np.array([], np.float32)
        return (np.vstack(rows).astype(np.float32),
                np.array(yx, np.float32),
                np.array(yy, np.float32))

    X_gbm_train, y_gbm_x_train, y_gbm_y_train = flatten_sequences_for_gbm(X_train, y_train, meta_train)
    X_gbm_val,   y_gbm_x_val,   y_gbm_y_val   = flatten_sequences_for_gbm(X_val,   y_val,   meta_val)

    if X_gbm_train.size == 0 or X_gbm_val.size == 0:
        print("⚠️  Flattened feature matrices are empty; skipping LightGBM and GRU training.")
        return

    print("Preparing to train LightGBM models...")
    if USE_TRAINED_MODELS and all(os.path.exists(f) for f in LGBM_X_FILES + LGBM_Y_FILES):
        print("⚡ Using pre-trained LightGBM models.")
        gbm_models_x = [joblib.load(f) for f in LGBM_X_FILES]
        gbm_models_y = [joblib.load(f) for f in LGBM_Y_FILES]
    else:
        def train_lgbm_regressor(X_train, y_train, X_val, y_val, name, seed):
            params = {
                "objective": "regression",
                "metric": "rmse",
                "learning_rate": 0.05,
                "num_leaves": 64,
                "max_depth": -1,
                "feature_fraction": 0.8,
                "bagging_fraction": 0.8,
                "bagging_freq": 5,
                "min_data_in_leaf": 20,
                "lambda_l1": 0.1, "lambda_l2": 0.1,
                "verbosity": -1, "seed": seed, "n_jobs": -1
            }
            dtrain = lgb.Dataset(X_train, label=y_train)
            dval = lgb.Dataset(X_val, label=y_val)
            model = lgb.train(
                params, dtrain, valid_sets=[dtrain, dval],
                valid_names=["train", "val"],
                num_boost_round=2000,
                callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
            )
            joblib.dump(model, f"{LGBM_DIR}/lgbm_{name}_seed{seed}.pkl")
            return model

        LGBM_SEEDS = [SEED, SEED + 1, SEED + 2]
        gbm_models_x, gbm_models_y = [], []
        for s in LGBM_SEEDS:
            print(f"Training LightGBM (X) seed={s}")
            gbm_models_x.append(train_lgbm_regressor(X_gbm_train, y_gbm_x_train, X_gbm_val, y_gbm_x_val, "x", s))
            print(f"Training LightGBM (Y) seed={s}")
            gbm_models_y.append(train_lgbm_regressor(X_gbm_train, y_gbm_y_train, X_gbm_val, y_gbm_y_val, "y", s))
        print("✅ LightGBM models trained & saved.")

    print(f"GBM models loaded: X={len(gbm_models_x)}, Y={len(gbm_models_y)}")
    print("🚀 Training GRU model (variable-length, fp32 loss)...")

    from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

    class GRURegressor(nn.Module):
        def __init__(self, input_dim, hidden_dim=GRU_HIDDEN):
            super().__init__()
            self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, 2)
        def forward(self, x, lengths):
            packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out_packed, _ = self.gru(packed)
            out, _ = pad_packed_sequence(out_packed, batch_first=True)
            out = self.fc(out)
            return out

    class SeqDataset(Dataset):
        def __init__(self, X, y): self.X, self.y = X, y
        def __len__(self): return len(self.X)
        def __getitem__(self, i):
            return torch.tensor(self.X[i],dtype=torch.float32), torch.tensor(self.y[i],dtype=torch.float32)

    def collate_pad(batch):
        xs, ys = zip(*batch)
        x_tensors = [torch.tensor(x,dtype=torch.float32) for x in xs]
        y_tensors = [torch.tensor(y,dtype=torch.float32) for y in ys]
        lengths = torch.tensor([x.shape[0] for x in x_tensors],dtype=torch.long)
        return pad_sequence(x_tensors,batch_first=True), pad_sequence(y_tensors,batch_first=True), lengths

    def masked_mse(pred, target, lengths):
        pred = torch.nan_to_num(pred, nan=0.0, posinf=1e3, neginf=-1e3).float()
        target = torch.nan_to_num(target, nan=0.0, posinf=1e3, neginf=-1e3).float()
        B, T, _ = pred.shape
        mask = (torch.arange(T, device=pred.device)[None, :] < lengths[:, None]).float().unsqueeze(-1)
        se = (pred - target).pow(2) * mask
        return se.sum() / mask.sum().clamp_min(1)

    train_ds, val_ds = SeqDataset(X_train, y_train), SeqDataset(X_val, y_val)
    train_dl = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True,num_workers=NUM_WORKERS,collate_fn=collate_pad)
    val_dl   = DataLoader(val_ds,batch_size=BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,collate_fn=collate_pad)

    if not X_train:
        print("⚠️  No sequences available for GRU training; skipping.")
        return

    model_gru = GRURegressor(X_train[0].shape[1]).to(device)
    for n,p in model_gru.named_parameters():
        if "weight_hh" in n and p.dim() >= 2: nn.init.orthogonal_(p)
    optimizer = torch.optim.AdamW(model_gru.parameters(), lr=GRU_LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6)
    scaler = GradScaler()

    best_rmse = float("inf")
    for epoch in range(1, MAX_EPOCHS+1):
        model_gru.train(); total_loss = 0; seen = 0
        for xb,yb,lengths in train_dl:
            xb,yb,lengths = xb.to(device),yb.to(device),lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model_gru(xb, lengths)
            loss = masked_mse(pred,yb,lengths)
            if not torch.isfinite(loss):
                print("⚠️  Skipping non-finite batch."); continue
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model_gru.parameters(), CLIP_VALUE)
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item()*xb.size(0); seen += xb.size(0)
        train_rmse = math.sqrt(max(total_loss,0)/max(seen,1))

        model_gru.eval(); val_loss=0; seen_v=0
        with torch.no_grad():
            for xb,yb,lengths in val_dl:
                xb,yb,lengths = xb.to(device),yb.to(device),lengths.to(device)
                pred = model_gru(xb, lengths)
                l = masked_mse(pred,yb,lengths)
                if not torch.isfinite(l): continue
                val_loss += l.item()*xb.size(0); seen_v += xb.size(0)
        val_rmse = math.sqrt(max(val_loss,0)/max(seen_v,1))
        print(f"Epoch {epoch:03d}: train_RMSE={train_rmse:.4f}, val_RMSE={val_rmse:.4f}")

        if math.isfinite(val_rmse) and val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save(model_gru.state_dict(), MODEL_CHECKPOINT)
            print(f"✅ Best val RMSE improved to {val_rmse:.5f}")
        scheduler.step(val_rmse)

    print(f"GRU training finished. Best Val RMSE: {best_rmse:.5f}")

    runtime = time.time()-start_total
    print("============================================================")
    print("MoveCast+ — Offline Training Finished")
    print("============================================================")
    print(f"Runtime: {format_runtime(runtime)}")
    print("============================================================")

# ============================================================
# Online (Evaluation API) inference
# ============================================================
_INFERENCE_INITIALIZED = False
_INFERENCE_ARTIFACTS = {"gbm_x": [], "gbm_y": [], "gru": None}


def _initialize_inference_artifacts():
    global _INFERENCE_INITIALIZED
    if _INFERENCE_INITIALIZED:
        return

    if USE_TRAINED_MODELS:
        try:
            _INFERENCE_ARTIFACTS["gbm_x"] = [joblib.load(f) for f in LGBM_X_FILES]
            _INFERENCE_ARTIFACTS["gbm_y"] = [joblib.load(f) for f in LGBM_Y_FILES]
            if GRU_CHECKPOINT_FILE:
                pass
            print("Loaded trained artifacts for inference.")
        except Exception as exc:
            print("Could not load trained artifacts; defaulting to kinematic predictions.", exc)
            _INFERENCE_ARTIFACTS["gbm_x"], _INFERENCE_ARTIFACTS["gbm_y"], _INFERENCE_ARTIFACTS["gru"] = [], [], None

    _INFERENCE_INITIALIZED = True


def predict(test: pl.DataFrame | pd.DataFrame, test_input: pl.DataFrame | pd.DataFrame) -> pl.DataFrame | pd.DataFrame:
    """Entry point used by the Kaggle inference server."""
    test_pd = ensure_pandas(test)
    test_input_pd = ensure_pandas(test_input)

    _initialize_inference_artifacts()

    required_cols = ["game_id", "play_id", "nfl_id", "frame_id"]
    missing_cols = [c for c in required_cols if c not in test_pd.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in prediction template: {missing_cols}")

    kinematic_full = kinematic_predict(test_input_pd, test_pd[required_cols])
    kinematic_full = kinematic_full.set_index(required_cols)
    target_index = pd.MultiIndex.from_frame(test_pd[required_cols])
    ordered = kinematic_full.reindex(target_index)
    ordered = ordered.fillna(0.0)
    result = ordered.reset_index(drop=True)[["x", "y"]]

    assert isinstance(result, pd.DataFrame)
    assert len(result) == len(test_pd)

    return result


def _make_inference_server():
    try:
        import kaggle_evaluation.nfl_inference_server as nfl_inference_server
    except ModuleNotFoundError as exc:  # pragma: no cover - local environments may lack the package
        print("kaggle_evaluation package not available; inference server cannot be created.", exc)
        return None

    return nfl_inference_server.NFLInferenceServer(predict)


INFERENCE_SERVER = None if DO_TRAIN else _make_inference_server()


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    if DO_TRAIN:
        offline_training_pipeline()
    else:
        if INFERENCE_SERVER is None:
            raise RuntimeError("Inference server failed to initialize.")
        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            INFERENCE_SERVER.serve()
        else:
            INFERENCE_SERVER.run_local_gateway(('/kaggle/input/nfl-big-data-bowl-2026-prediction/',))
