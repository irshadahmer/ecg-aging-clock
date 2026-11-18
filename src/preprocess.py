import os
import ast
import wfdb
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from tqdm import tqdm

raw_dir = os.path.join("data", "raw", "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3")
out_dir = os.path.join("data", "processed")
os.makedirs(out_dir, exist_ok=True)

def bandpass_filter(sig, fs=100.0, low=0.5, high=40.0):
    nyq = 0.5 * fs
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, sig, axis=-1)

def zscore_normalize(sig):
    mean = np.mean(sig, axis=-1, keepdims=True)
    std = np.std(sig, axis=-1, keepdims=True) + 1e-8
    return ((sig - mean) / std).astype(np.float32)

df = pd.read_csv(os.path.join(raw_dir, "ptbxl_database.csv"), index_col="ecg_id")
df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)

scp_df = pd.read_csv(os.path.join(raw_dir, "scp_statements.csv"), index_col=0)
scp_df = scp_df[scp_df["diagnostic"] == 1]

def get_superclasses(code_dict):
    classes = set()
    for code in code_dict:
        if code in scp_df.index:
            classes.add(scp_df.loc[code, "diagnostic_class"])
    return list(classes)

df["superclasses"] = df["scp_codes"].apply(get_superclasses)
df = df[(df["age"] >= 18) & (df["age"] <= 95)].copy()
df["sex"] = df["sex"].fillna(0).astype(int)

for sc in ["NORM", "MI", "STTC", "CD", "HYP"]:
    df[f"diag_{sc}"] = df["superclasses"].apply(lambda x: int(sc in x))

signals = []
valid_ids = []

for ecg_id, row in tqdm(df.iterrows(), total=len(df), desc="Processing ECGs"):
    rec_path = os.path.join(raw_dir, row["filename_lr"])
    try:
        sig, _ = wfdb.rdsamp(rec_path)
        if sig.shape != (1000, 12) or np.isnan(sig).any():
            continue
        sig = bandpass_filter(sig.T)
        sig = zscore_normalize(sig)
        signals.append(sig)
        valid_ids.append(ecg_id)
    except Exception:
        continue

meta = df.loc[valid_ids].reset_index()
meta["split"] = np.where(meta["strat_fold"] <= 8, "train", np.where(meta["strat_fold"] == 9, "val", "test"))

X = np.stack(signals, axis=0)
np.save(os.path.join(out_dir, "X_ecg.npy"), X)
meta.to_parquet(os.path.join(out_dir, "meta.parquet"), index=False)

print(f"Processed {X.shape[0]} ECGs with shape {X.shape}.")
print(meta["split"].value_counts())
