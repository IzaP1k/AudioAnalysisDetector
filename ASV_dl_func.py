import copy
import json
import os
import pickle
import random
import time
import numpy as np
import pandas as pd
import soundfile as sf
import joblib
from joblib import Parallel, delayed
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from prepare_data import prepare_filepaths

import torch
import matplotlib.pyplot as plt

from classic_ml_func import GMMClassifier, BiLSTMClassifier, compute_eer
from prepare_data import detect_columns
from pytorch_func import FeatureColumnDataset, AntiSpoofingResNet
from visualisation_func import model_result_metrics

try:
    import shap
except Exception:
    shap = None
try:
    from lime.lime_tabular import LimeTabularExplainer
except Exception:
    LimeTabularExplainer = None

from omegaconf import OmegaConf


config = OmegaConf.load("config.yaml")

METADATA_PATH_DF = config.datasets.DF.metadata
FLAC_FOLDER_DF_1 = config.datasets.DF.flac[0]
FLAC_FOLDER_DF_2 = config.datasets.DF.flac[1]
COLS_DF = config.datasets.DF.columns

METADATA_PATH_PA = config.datasets.PA.metadata
FLAC_FOLDER_PA_1 = config.datasets.PA.flac[0]
FLAC_FOLDER_PA_2 = config.datasets.PA.flac[1]
COLS_PA = config.datasets.PA.columns

METADATA_PATH_LA = config.datasets.LA.metadata
FLAC_FOLDER_LA_1 = config.datasets.LA.flac[0]
COLS_LA = config.datasets.LA.columns


def add_dataAugmentation(df, col_name="augmentationType", aug_type=None):
    if aug_type is None:
        aug_type = ['change pitch', 'noise']

    if col_name not in df.columns:
        df[col_name] = None
    else:
        df[col_name] = None

    extra_rows = []

    for _, row in df.iterrows():
        if random.random() < 0.8:
            chosen_aug = random.choice(aug_type)
            row_copy = row.copy()
            row_copy[col_name] = chosen_aug
            extra_rows.append(row_copy)

        if random.random() < 0.5 and len(aug_type) > 1:
            aug_pair = random.sample(aug_type, 2)
            for aug in aug_pair:
                row_copy = row.copy()
                row_copy[col_name] = aug
                extra_rows.append(row_copy)

    if extra_rows:
        df_aug = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)
    else:
        print("Brak zmian")
        df_aug = df.copy()

    return df_aug

def downsampled_dataset(df, label1=1, label2=0):
    class_1 = df[df.target == label1]
    class_2 = df[df.target == label2]

    if class_1 < class_2:
        minority_class = class_1
        majority_class = class_2

    else:
        minority_class = class_2
        majority_class = class_1

    majority_downsampled = resample(majority_class,
                                    replace=False,
                                    n_samples=len(minority_class),
                                    random_state=42)

    df_balanced = pd.concat([majority_downsampled, minority_class])

    return df_balanced


def run_train_step(model, loader, optimizer, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)

        if outputs.ndim == 2 and outputs.shape[1] == 1:
            loss = criterion(outputs, y_batch.float())
            predicted = (torch.sigmoid(outputs) >= 0.5).float()
        else:
            loss = criterion(outputs, y_batch.long())
            _, predicted = torch.max(outputs, 1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X_batch.size(0)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    return running_loss / total, correct / total


def run_val_step(model, loader, criterion, device):
    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)

            if outputs.ndim == 2 and outputs.shape[1] == 1:
                loss = criterion(outputs, y_batch.float())
                predicted = (torch.sigmoid(outputs) >= 0.5).float()
            else:
                loss = criterion(outputs, y_batch.long())
                _, predicted = torch.max(outputs, 1)

            val_loss += loss.item() * X_batch.size(0)
            val_total += y_batch.size(0)
            val_correct += (predicted == y_batch).sum().item()

    return val_loss / val_total, val_correct / val_total


def plot_loss_curves(train_losses, val_losses, feature_col):
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Treningowe wartości straty')
    plt.plot(val_losses, label='Walidacyjne wartości straty')
    plt.title(f'Krzywa wartości straty w trakcie epok nauki dla cechy {feature_col}')
    plt.xlabel('Epoka')
    plt.ylabel('Wartość straty')
    plt.legend()
    plt.grid(True)
    plt.show()


def train_loop(model, optimizer, criterion, train_loader, test_loader, device,
               train_losses, val_losses, feature_col, epochs=100):
    logs = []
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        train_loss, train_acc = run_train_step(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = run_val_step(model, test_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        log_entry = (f"[{feature_col}] Epoch {epoch + 1}/{epochs} | "
                     f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                     f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        logs.append(log_entry)
        print(log_entry)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())

    plot_loss_curves(train_losses, val_losses, feature_col)

    return copy.deepcopy(model.state_dict()), best_model_state, logs


def prepare_dataloaders(final_df, test_df, feature_col, label_col, batch_size):
    if test_df is None:
        X_train_df, X_test_df = train_test_split(final_df, test_size=0.2,
                                                 stratify=final_df[label_col], random_state=42)
    else:
        X_train_df = final_df
        X_test_df = test_df

    train_dataset = FeatureColumnDataset(X_train_df, feature_col, label_col)
    test_dataset = FeatureColumnDataset(X_test_df, feature_col, label_col)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def evaluate_final_models(model, best_model_state, oldest_model_state, test_loader, device):
    oldest_model = copy.deepcopy(model)
    oldest_model.load_state_dict(oldest_model_state)

    best_model = copy.deepcopy(model)
    best_model.load_state_dict(best_model_state)

    y_true_best, y_pred_best, y_scores_best = [], [], []
    best_metrics = model_result_metrics(best_model, test_loader, device, y_true_best, y_pred_best, y_scores_best)

    y_true_old, y_pred_old, y_scores_old = [], [], []
    oldest_metrics = model_result_metrics(oldest_model, test_loader, device, y_true_old, y_pred_old, y_scores_old)

    return best_model, oldest_model, best_metrics, oldest_metrics


def train_feature_model(model, final_df, feature_col, label_col='label', batch_size=32, epochs=10,
                        device=None, test_df=None, criterion=None, optimizer=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader = prepare_dataloaders(final_df, test_df, feature_col, label_col, batch_size)

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    train_losses, val_losses = [], []

    oldest_model_state, best_model_state, logs = train_loop(
        model, optimizer, criterion, train_loader, test_loader,
        device, train_losses, val_losses, feature_col, epochs
    )

    best_model, oldest_model, best_metrics, oldest_metrics = evaluate_final_models(
        model, best_model_state, oldest_model_state, test_loader, device
    )

    return {
        "best_model": best_model,
        "oldest_model": oldest_model,
        "test_loader": test_loader,
        "logs": logs,
        "best_metrics": best_metrics,
        "oldest_metrics": oldest_metrics
    }


def process_scaling(final_df, test_df, feat, feat_dir):
    scaler = StandardScaler()
    all_train_features_for_scaler = np.vstack(final_df[feat].values)
    scaler.fit(all_train_features_for_scaler)

    final_df[feat] = final_df[feat].apply(lambda x: scaler.transform(x))
    if test_df is not None and not test_df.empty:
        test_df[feat] = test_df[feat].apply(lambda x: scaler.transform(x))

    scaler_path = os.path.join(feat_dir, f"{feat}_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    return scaler_path


def save_artifacts(feat_dir, feat, results, opt_name, loss_name, scaler_path):
    best_model_path = os.path.join(feat_dir, f"{feat}_best_model.pt")
    oldest_model_path = os.path.join(feat_dir, f"{feat}_oldest_model.pt")
    
    torch.save(results["best_model"].state_dict(), best_model_path)
    torch.save(results["oldest_model"].state_dict(), oldest_model_path)

    logs_path = os.path.join(feat_dir, f"{feat}_logs.json")
    with open(logs_path, "w", encoding="utf-8") as f:
        json.dump(results["logs"], f, indent=4, ensure_ascii=False)

    metrics_path = os.path.join(feat_dir, f"{feat}_metrics.json")
    metrics_data = {
        "optimizer": opt_name,
        "criterion": loss_name,
        "feature": feat,
        "best_metrics": results["best_metrics"],
        "oldest_metrics": results["oldest_metrics"]
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4, ensure_ascii=False)

    return {
        "best_model": best_model_path,
        "oldest_model": oldest_model_path,
        "metrics": metrics_path,
        "logs": logs_path,
        "scaler": scaler_path
    }


def train_all_features(final_df, feature_cols, test_df=None, label_col='label',
                       batch_size=32, epochs=10, model_dir="Res_Net", standard_scaler=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_models = {}
    os.makedirs(model_dir, exist_ok=True)

    optimizers_list = {
        "AdamW": torch.optim.AdamW,
        "SGD": torch.optim.SGD
    }
    loss_functions = {
        "CrossEntropyLoss": nn.CrossEntropyLoss,
    }

    for opt_name, opt_class in optimizers_list.items():
        for loss_name, loss_class in loss_functions.items():
            combo_name = f"{opt_name}_{loss_name}"
            print(f"\n========== Trening z {combo_name} ==========")
            combo_dir = os.path.join(model_dir, combo_name)
            os.makedirs(combo_dir, exist_ok=True)

            for feat in feature_cols:
                print(f"\n=== TRENING dla cechy: {feat} ===")

                if final_df.empty:
                    print(f"[UWAGA] Brak danych do treningu dla cechy {feat}!")
                    continue

                feat_dir = os.path.join(combo_dir, feat)
                os.makedirs(feat_dir, exist_ok=True)

                scaler_path = None
                if standard_scaler:
                    scaler_path = process_scaling(final_df, test_df, feat, feat_dir)

                model = AntiSpoofingResNet(num_classes=2).to(device)
                criterion = loss_class()
                
                if opt_name == "SGD":
                    optimizer = opt_class(model.parameters(), lr=1e-3, momentum=0.9)
                else:
                    optimizer = opt_class(model.parameters(), lr=1e-4, weight_decay=1e-5)

                results = train_feature_model(
                    model, final_df, feat, label_col, batch_size, epochs,
                    device, test_df=test_df, optimizer=optimizer, criterion=criterion
                )

                saved_paths = save_artifacts(feat_dir, feat, results, opt_name, loss_name, scaler_path)
                trained_models[(feat, combo_name)] = saved_paths

                print(f"Zapisano modele i wyniki dla '{feat}' → {combo_name}")
                print(f"Folder: {feat_dir}")
                print(f"=== KONIEC treningu dla cechy: {feat} ===\n")

            print(f"========== KONIEC treningu dla {combo_name} ==========\n")

    return trained_models

def extract_features(final_df, feature_extractors_map, col_name='filepath', mean=False, aug_col="augmentationType"):

    for name, func in feature_extractors_map.items():
        print(f"   - Ekstrahuję: {name}")

        results = Parallel(n_jobs=-1, verbose=1)(
            delayed(func)(
                row[col_name],
                chunk_start=row.get('chunk_start', None),
                chunk_end=row.get('chunk_end', None),
                mean=mean,
                augment=row.get(aug_col, None)
            )
            for _, row in final_df.iterrows()
        )

        final_df[name] = results

    return final_df


def transpose_cqcc(x):
    arr = np.array(x)
    if arr.ndim == 1:
        return arr[:, np.newaxis]
    elif arr.ndim == 2:
        if arr.shape[0] < arr.shape[1]:
            return arr.T
        else:
            return arr
    else:
        return None


def filtr_nan(final_df, col_name="cqcc"):
    initial_len = len(final_df)
    final_df = final_df[final_df[col_name].notnull()]
    if len(final_df) < initial_len:
        print(f"Usunięto {initial_len - len(final_df)} wierszy z pustymi wartościami {col_name}.")

    return final_df


def balance_func(final_df, col_name='label_num'):
    df_genuine = final_df[final_df[col_name] == 0]
    df_df = final_df[final_df[col_name] == 1]

    if len(df_genuine) > len(df_df):
        df_df_upsampled = resample(df_df, replace=True, n_samples=len(df_genuine), random_state=42)
        final_df_balanced = pd.concat([df_genuine, df_df_upsampled])
    else:
        df_genuine_upsampled = resample(df_genuine, replace=True, n_samples=len(df_df), random_state=42)
        final_df_balanced = pd.concat([df_genuine_upsampled, df_df])

    print(
        f"Zbilansowane dane: true={len(final_df_balanced[final_df_balanced[col_name] == 0])}, false={len(final_df_balanced[final_df_balanced[col_name] == 1])}")

    return final_df_balanced


def prepare_train_test_data(df, test_df=None, col_name="cqcc", label_name="label_num", model_dir="GMM-BiLSTM"):
    if test_df is None:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[label_name])
    else:
        train_df = df

    scaler = StandardScaler()

    all_train_features_for_scaler = np.vstack(train_df[col_name].values)
    scaler.fit(all_train_features_for_scaler)

    train_df[col_name] = train_df[col_name].apply(lambda x: scaler.transform(x))
    test_df[col_name] = test_df[col_name].apply(lambda x: scaler.transform(x))

    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

    return train_df, test_df, scaler

