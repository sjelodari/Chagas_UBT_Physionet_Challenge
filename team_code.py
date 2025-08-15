#!/usr/bin/env python

# team_code.py for PhysioNet Challenge 2025 - Chagas Disease Detection
# Author: Saber Jelodari
# Date 15 Aug 2025
# Version: V2-2

import os
import csv
import random
import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import scipy.signal as sgn
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

from helper_code import (
    find_records,
    load_signals,
    load_label,
    load_header,
    load_source,
    get_age,
    get_sex,
)


# Hyperparameters & Config
HP = {
    # CNN parameters
    "cnn_channels": [12, 32, 64, 128],
    "dropout_rate": 0.4,
    # LSTM parameters
    "rnn_hidden_size": 128,
    "rnn_num_layers": 2,

    # Attention parameters
    "attention_size": 128,

    # Fully-connected parameters
    "fc_size": 128,

    # Data parameters
    "desired_len": 4000,
    "random_crop": True,

    # Training parameters
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 40,
    "weight_decay": 1e-5,

    # Focal Loss parameters (replaces pos_weight)
    "focal_loss_alpha": 0.25,
    "focal_loss_gamma": 2.0,

    # Data augmentation probabilities
    "augment_time_shift": 0.5,
    "max_time_shift": 200,
    "augment_amp_scaling": 0.5,
    "amp_scale_range": [0.9, 1.1],
    "augment_lead_dropout": 0.1,
    "augment_noise_prob": 0.5,
    "noise_level": 0.01,
    "use_mixup": True,
    "mixup_alpha": 0.4,

    # Early stopping
    "early_stopping_patience": 7,

    # Decision threshold
    "decision_threshold": 0.5,

    # Weighted Sampler config
    "strong_label_weight_multiplier": 2.0,
}

# Will store the best threshold found by the training loop for final inference
BEST_THRESHOLD = 0.5


# Device selection

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# -Focal Loss Implementation ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_with_logits = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce_with_logits(inputs, targets)
        p_t = torch.exp(-bce_loss)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_loss = alpha_t * (1 - p_t)**self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# --Attention Layer ---
class Attention(nn.Module):
    def __init__(self, feature_dim, step_dim, bias=True, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.feature_dim = feature_dim
        self.step_dim = step_dim
        self.bias = bias

        self.e = nn.Linear(feature_dim, HP["attention_size"], bias=bias)
        self.t = nn.Linear(HP["attention_size"], 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x shape: [batch_size, seq_len, num_features]
        e_out = torch.tanh(self.e(x))
        t_out = self.t(e_out)
        a = self.softmax(t_out)
        context = torch.sum(a * x, dim=1)
        return context

# Model Definition
class ChagasModel(nn.Module):
    def __init__(self):
        super().__init__()

        # -------------------- CNN Layers --------------------
        self.conv1 = nn.Conv1d(HP["cnn_channels"][0], HP["cnn_channels"][1], kernel_size=5, padding=2)
        self.bn1   = nn.BatchNorm1d(HP["cnn_channels"][1])
        self.conv2 = nn.Conv1d(HP["cnn_channels"][1], HP["cnn_channels"][2], kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(HP["cnn_channels"][2])
        self.conv3 = nn.Conv1d(HP["cnn_channels"][2], HP["cnn_channels"][3], kernel_size=5, padding=2)
        self.bn3   = nn.BatchNorm1d(HP["cnn_channels"][3])

        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(HP["dropout_rate"])
        self.adaptive_pool = nn.AdaptiveAvgPool1d(100)

        # -------------------- LSTM Layer --------------------
        self.lstm = nn.LSTM(
            input_size=HP["cnn_channels"][3],
            hidden_size=HP["rnn_hidden_size"],
            num_layers=HP["rnn_num_layers"],
            batch_first=True,
            bidirectional=True,
            dropout=HP["dropout_rate"] if HP["rnn_num_layers"] > 1 else 0
        )

        # --Use Attention layer ---
        self.attention = Attention(HP["rnn_hidden_size"] * 2, 100)

        # -------------------- FC Layers ---------------------
        self.fc1 = nn.Linear(2 * HP["rnn_hidden_size"], HP["fc_size"])
        self.fc2 = nn.Linear(HP["fc_size"], 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout(x)

        x = self.adaptive_pool(x)
        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)

        # --- Use Attention instead of mean pooling ---
        attention_out = self.attention(lstm_out)

        x = F.relu(self.fc1(attention_out))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


# Data Processing / Resampling / Augmentation
def resample_to_400Hz(signals, original_fs):
    if original_fs == 0: original_fs = 400
    if original_fs == 400: return signals
    new_len = int(signals.shape[0] * 400 / original_fs)
    return sgn.resample(signals, new_len, axis=0)

def random_time_shift(ecg):
    max_shift = HP["max_time_shift"]
    shift = random.randint(-max_shift, max_shift)
    return torch.roll(ecg, shifts=shift, dims=1)

def random_amplitude_scaling(ecg):
    scale = random.uniform(*HP["amp_scale_range"])
    return ecg * scale

def random_lead_dropout(ecg):
    lead_to_drop = random.randint(0, ecg.shape[0] - 1)
    ecg[lead_to_drop, :] = 0.0
    return ecg

def add_gaussian_noise(ecg):
    noise = torch.randn_like(ecg) * HP["noise_level"]
    return ecg + noise

def process_ecg(signals, fields, augment=False):
    fs = fields.get('fs', 400)
    signals = resample_to_400Hz(signals, fs)

    if signals.shape[1] < 12:
        tmp = np.zeros((signals.shape[0], 12), dtype=signals.dtype)
        tmp[:, :signals.shape[1]] = signals
        signals = tmp
    elif signals.shape[1] > 12:
        signals = signals[:, :12]

    ecg = torch.tensor(signals.T, dtype=torch.float32)

    desired_len = HP["desired_len"]
    cur_len = ecg.shape[1]
    if cur_len > desired_len:
        if HP["random_crop"] and augment:
            start = random.randint(0, cur_len - desired_len)
        else:
            start = (cur_len - desired_len) // 2
        ecg = ecg[:, start:start + desired_len]
    elif cur_len < desired_len:
        pad = torch.zeros((12, desired_len))
        pad[:, :cur_len] = ecg
        ecg = pad

    if augment:
        if random.random() < HP["augment_time_shift"]: ecg = random_time_shift(ecg)
        if random.random() < HP["augment_amp_scaling"]: ecg = random_amplitude_scaling(ecg)
        if random.random() < HP["augment_lead_dropout"]: ecg = random_lead_dropout(ecg)
        if random.random() < HP["augment_noise_prob"]: ecg = add_gaussian_noise(ecg) # <-- New

    mean = ecg.mean(dim=1, keepdim=True)
    std = ecg.std(dim=1, keepdim=True) + 1e-6
    ecg = (ecg - mean) / std
    return ecg

# --- Mixup Implementation ---
def mixup_data(x, y, alpha=1.0, device='cuda'):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def calculate_metrics(y_true, y_pred, y_prob):
    metrics = {}
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
    metrics['sensitivity'] = recall_score(y_true, y_pred, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
    try:
        metrics['auc'] = roc_auc_score(y_true, y_prob)
    except:
        metrics['auc'] = 0.5
    return metrics

# - PyTorch Dataset Class ---
class ECGDataset(Dataset):
    def __init__(self, records, augment=False):
        self.records = records
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec_path, label, _ = self.records[idx]
        try:
            signals, fields = load_signals(rec_path)
            ecg = process_ecg(signals, fields, augment=self.augment)
            return ecg, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            # On error, return a dummy tensor.
            print(f"Warning: Could not load {rec_path}. Error: {e}. Returning dummy data.")
            return torch.zeros((12, HP["desired_len"]), dtype=torch.float32), torch.tensor(0.0, dtype=torch.float32)


# Required Challenge Functions
def train_model(data_folder, model_folder, verbose):
    global BEST_THRESHOLD
    device = get_device()
    if verbose: print(f"[train_model] Using device: {device}")

    records = find_records(data_folder)
    if verbose: print(f"[train_model] Found {len(records)} records.")

    # Collect valid records with labels AND source for weighted sampling
    valid_records = []
    for record in tqdm(records, desc="Loading record metadata"):
        record_path = os.path.join(data_folder, record)
        try:
            # --Load source to distinguish weak/strong labels ---
            header = load_header(record_path)
            label = load_label(record_path)
            source = load_source(record_path)
            valid_records.append((record_path, label, source))
        except Exception as e:
            if verbose: print(f"Skipping {record}: {str(e)}")

    positives = sum(1 for _, lbl, _ in valid_records if lbl == 1)
    if verbose: print(f"Total valid records: {len(valid_records)}; Positives: {positives}, Negatives: {len(valid_records)-positives}")

    # -- Split before any balancing ---
    random.shuffle(valid_records)
    split_idx = int(0.8 * len(valid_records))
    train_records = valid_records[:split_idx]
    val_records = valid_records[split_idx:]
    if verbose: print(f"Training set: {len(train_records)}, Validation set: {len(val_records)}")

    train_dataset = ECGDataset(train_records, augment=True)
    val_dataset = ECGDataset(val_records, augment=False)

    # - Implement Weighted Sampling for training data ---
    train_labels = [rec[1] for rec in train_records]
    class_counts = np.bincount(train_labels)
    num_neg, num_pos = class_counts[0], class_counts[1]

    # Base weight on class imbalance
    class_weights = [1.0 / num_neg, 1.0 / num_pos]
    sample_weights = [class_weights[lbl] for lbl in train_labels]

    # Adjust weights based on label quality (strong vs weak)
    strong_labels = {'SaMi-Trop', 'PTB-XL'}
    for i, (_, _, source) in enumerate(train_records):
        if source in strong_labels:
            sample_weights[i] *= HP["strong_label_weight_multiplier"]

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=HP["batch_size"], sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=HP["batch_size"], shuffle=False)

    model = ChagasModel().to(device)
    criterion = FocalLoss(alpha=HP["focal_loss_alpha"], gamma=HP["focal_loss_gamma"]) # Use Focal Loss
    optimizer = optim.Adam(model.parameters(), lr=HP["learning_rate"], weight_decay=HP["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0
    os.makedirs(model_folder, exist_ok=True) # Ensure model folder exists

    # --- NEW: Initialize CSV file for metrics logging ---
    metrics_log_path = os.path.join(model_folder, "training_metrics.csv")
    csv_headers = [
        "epoch", "train_loss", "val_loss", "f1", "auc",
        "sensitivity", "precision", "accuracy", "best_threshold"
    ]
    with open(metrics_log_path, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_headers)

        for epoch in range(1, HP["epochs"] + 1):
            model.train()
            running_loss = 0.0
            for ecgs_t, labels_t in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
                ecgs_t, labels_t = ecgs_t.to(device), labels_t.to(device).view(-1, 1)

                # --- NEW: Apply Mixup ---
                if HP["use_mixup"] and HP["mixup_alpha"] > 0:
                    mixed_ecgs, labels_a, labels_b, lam = mixup_data(ecgs_t, labels_t, HP["mixup_alpha"], device)
                    logits = model(mixed_ecgs)
                    loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
                else:
                    logits = model(ecgs_t)
                    loss = criterion(logits, labels_t)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * ecgs_t.size(0)

            avg_train_loss = running_loss / len(train_loader.sampler)

            # Validation
            model.eval()
            val_loss = 0.0
            y_true, y_scores = [], []
            with torch.no_grad():
                for ecgs_t, labels_t in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
                    ecgs_t, labels_t = ecgs_t.to(device), labels_t.to(device).view(-1, 1)
                    logits = model(ecgs_t)
                    val_loss += criterion(logits, labels_t).item() * ecgs_t.size(0)

                    prob = torch.sigmoid(logits).cpu().numpy().flatten()
                    y_true.extend(labels_t.cpu().numpy().flatten())
                    y_scores.extend(prob)

            avg_val_loss = val_loss / len(val_loader.dataset) if val_loader.dataset else 0
            scheduler.step(avg_val_loss) # LR scheduler step

            best_f1_for_epoch, best_thresh_for_epoch = -1, HP["decision_threshold"]
            if y_true:
                for t in np.linspace(0.1, 0.9, 81):
                    f1_t = f1_score(y_true, [1 if s >= t else 0 for s in y_scores], zero_division=0)
                    if f1_t > best_f1_for_epoch:
                        best_f1_for_epoch = f1_t
                        best_thresh_for_epoch = t

            preds_val = [1 if s >= best_thresh_for_epoch else 0 for s in y_scores]
            metrics_val = calculate_metrics(y_true, preds_val, y_scores)

            if verbose:
                print(f"Epoch {epoch}/{HP['epochs']} | TrainLoss={avg_train_loss:.4f} | ValLoss={avg_val_loss:.4f} | "
                      f"F1={metrics_val['f1']:.4f} | AUC={metrics_val['auc']:.4f} | BestThresh={best_thresh_for_epoch:.3f}")

            # - Log metrics for the current epoch to the CSV file ---
            row_data = [
                epoch,
                round(avg_train_loss, 5),
                round(avg_val_loss, 5),
                round(metrics_val['f1'], 5),
                round(metrics_val['auc'], 5),
                round(metrics_val['sensitivity'], 5),
                round(metrics_val['precision'], 5),
                round(metrics_val['accuracy'], 5),
                round(best_thresh_for_epoch, 3)
            ]
            writer.writerow(row_data)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                BEST_THRESHOLD = best_thresh_for_epoch
                epochs_no_improve = 0
                torch.save({'model_state': model.state_dict(), 'best_threshold': BEST_THRESHOLD},
                           os.path.join(model_folder, 'model.pth'))
                if verbose: print(f"Validation loss improved. Saving model with threshold {BEST_THRESHOLD:.3f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= HP["early_stopping_patience"]:
                    if verbose: print("Early stopping triggered.")
                    break

    if verbose:
        print(f"Training finished. Best threshold found: {BEST_THRESHOLD:.3f}")
        # where the log file is saved ---
        print(f"Metrics saved to {metrics_log_path}")

def load_model(model_folder, verbose):
    global BEST_THRESHOLD
    device = get_device()
    model_path = os.path.join(model_folder, 'model.pth')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = ChagasModel().to(device)
    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    BEST_THRESHOLD = checkpoint.get('best_threshold', 0.5) # Load best threshold from checkpoint

    if verbose:
        print(f"Loaded PyTorch model successfully. Using threshold: {BEST_THRESHOLD:.3f}")

    return {'model': model}

def run_model(record, model_dict, verbose):
    global BEST_THRESHOLD
    device = get_device()
    model = model_dict['model']
    model.eval()
    model.to(device)

    try:
        signals, fields = load_signals(record)
    except Exception as e:
        if verbose: print(f"Error loading {record}: {e}")
        return 0, 0.01 # Return negative with low probability on error

    ecg = process_ecg(signals, fields, augment=False).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(ecg)
        prob = torch.sigmoid(logits).item()

    binary_output = int(prob >= BEST_THRESHOLD)
    probability_output = float(prob)

    return binary_output, probability_output

