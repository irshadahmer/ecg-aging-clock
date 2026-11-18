import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from model import ECGResNet1D

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

X = np.load(os.path.join("data", "processed", "X_ecg.npy"))
meta = pd.read_parquet(os.path.join("data", "processed", "meta.parquet"))

train_mask = (meta["split"] == "train").values
val_mask = (meta["split"] == "val").values
test_mask = (meta["split"] == "test").values

X_train = torch.tensor(X[train_mask], dtype=torch.float32)
y_train = torch.tensor(meta.loc[train_mask, "age"].values, dtype=torch.float32)

X_val = torch.tensor(X[val_mask], dtype=torch.float32)
y_val = torch.tensor(meta.loc[val_mask, "age"].values, dtype=torch.float32)

X_test = torch.tensor(X[test_mask], dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=128, shuffle=False)

def predict_batches(model, tensor_x):
    model.eval()
    preds = []
    loader = DataLoader(TensorDataset(tensor_x), batch_size=128, shuffle=False)
    with torch.no_grad():
        for (bx,) in loader:
            preds.append(model(bx.to(device)).cpu().numpy())
    return np.concatenate(preds)

def train_teacher(epochs=15):
    model = ECGResNet1D(in_channels=12, base_filters=32).to(device)
    criterion = nn.SmoothL1Loss(beta=5.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3, epochs=epochs, steps_per_epoch=len(train_loader)
    )

    best_mae = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                val_preds.append(model(bx.to(device)).cpu().numpy())
                val_targets.append(by.numpy())
        val_mae = np.mean(np.abs(np.concatenate(val_preds) - np.concatenate(val_targets)))
        print(f"[12-Lead Teacher] Epoch {epoch:02d}/{epochs} | Val MAE: {val_mae:.2f} yrs")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), os.path.join("results", "teacher_12lead.pt"))

    model.load_state_dict(torch.load(os.path.join("results", "teacher_12lead.pt"), map_location=device))
    return model

def train_student_distill(teacher, epochs=15):
    teacher.eval()
    with torch.no_grad():
        teacher_train_preds = predict_batches(teacher, X_train)
    teacher_train_tensor = torch.tensor(teacher_train_preds, dtype=torch.float32)

    s_train_loader = DataLoader(
        TensorDataset(X_train[:, 0:1, :], y_train, teacher_train_tensor), batch_size=64, shuffle=True
    )
    s_val_loader = DataLoader(TensorDataset(X_val[:, 0:1, :], y_val), batch_size=128, shuffle=False)

    student = ECGResNet1D(in_channels=1, base_filters=32).to(device)
    criterion = nn.SmoothL1Loss(beta=5.0)
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3, epochs=epochs, steps_per_epoch=len(s_train_loader)
    )

    best_mae = float("inf")
    for epoch in range(1, epochs + 1):
        student.train()
        for bx, by, bt in s_train_loader:
            bx, by, bt = bx.to(device), by.to(device), bt.to(device)
            optimizer.zero_grad()
            out = student(bx)
            loss = 0.5 * criterion(out, by) + 0.5 * criterion(out, bt)
            loss.backward()
            optimizer.step()
            scheduler.step()

        student.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, by in s_val_loader:
                val_preds.append(student(bx.to(device)).cpu().numpy())
                val_targets.append(by.numpy())
        val_mae = np.mean(np.abs(np.concatenate(val_preds) - np.concatenate(val_targets)))
        print(f"[Lead-I Student]  Epoch {epoch:02d}/{epochs} | Val MAE: {val_mae:.2f} yrs")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(student.state_dict(), os.path.join("results", "student_1lead.pt"))

    student.load_state_dict(torch.load(os.path.join("results", "student_1lead.pt"), map_location=device))
    return student

if __name__ == "__main__":
    teacher = train_teacher(epochs=15)
    student = train_student_distill(teacher, epochs=15)

    eval_meta = meta[meta["split"].isin(["val", "test"])].copy().reset_index(drop=True)
    eval_X = torch.tensor(X[meta["split"].isin(["val", "test"]).values], dtype=torch.float32)

    eval_meta["pred_age_12lead"] = predict_batches(teacher, eval_X)
    eval_meta["pred_age_1lead"] = predict_batches(student, eval_X[:, 0:1, :])

    eval_meta.to_parquet(os.path.join("results", "predictions.parquet"), index=False)
    print("Training complete. Saved results/predictions.parquet")
