
import os

import numpy as np
import pandas as pd
import librosa
import torch
from IPython.display import Audio, display
from matplotlib import pyplot as plt
import warnings

from sklearn.metrics import f1_score, accuracy_score, roc_curve

warnings.filterwarnings('ignore', category=UserWarning)

from omegaconf import OmegaConf


config = OmegaConf.load("config.yaml")
FLAC_FOLDER_LA_1 = config.datasets.LA.flac[0]

def prepare_filepath(df, file_id_col="file_id", flac=FLAC_FOLDER_LA_1):
    df["file_name"] = df[file_id_col] + ".flac"
    df["file_path"] = df["file_name"].apply(lambda x: os.path.join(flac, x))

    return df[df["file_path"].apply(os.path.exists)]

def plot_coeff_histograms_by_label_separately(df, coeff_col, label_col='label'):

    df_clean = df.dropna(subset=[coeff_col]).copy()
    if df_clean.empty:
        print(f"Brak danych dla cechy {coeff_col} do narysowania histogramu.")
        return

    n_coeffs = len(df_clean[coeff_col].iloc[0])
    mfcc_df = pd.DataFrame(df_clean[coeff_col].tolist(), columns=[f'{coeff_col}_{i + 1}' for i in range(n_coeffs)])
    df_full = pd.concat([df_clean[label_col].reset_index(drop=True), mfcc_df], axis=1)

    labels = df_full[label_col].unique()
    colors = dict(zip(labels, ['skyblue', 'salmon', 'lightgreen', 'plum']))

    for i in range(n_coeffs):
        col_name = f'{coeff_col}_{i + 1}'
        plt.figure(figsize=(6, 4))
        for label in labels:
            subset = df_full[df_full[label_col] == label][col_name]
            if not subset.dropna().empty:
                plt.hist(subset, bins=10, alpha=0.6, label=label, color=colors.get(label), edgecolor='black')
        plt.title(col_name)
        plt.xlabel('Wartość')
        plt.ylabel('Liczba wystąpień')
        plt.legend()
        plt.tight_layout()
        plt.show()

def listen_voice_flac(df, file_path_col="file_path", label_col="label"):

    samples = df.sample(5, random_state=42)[[file_path_col, label_col]].reset_index(drop=True)
    for i, row in samples.iterrows():
        print(f"{i + 1}. {row[label_col].upper()} — {os.path.basename(row[file_path_col])}")
        try:
            y, sr = librosa.load(row[file_path_col], sr=None)
            display(Audio(y, rate=sr))
        except Exception as e:
            print(f"  Nie można wczytać pliku: {e}")


def model_result_metrics(model, test_loader, device, y_true, y_pred, y_scores):
    model.eval()
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)

            if outputs.ndim == 2 and outputs.shape[1] == 1:
                probs = torch.sigmoid(outputs).squeeze(1)
                predicted = (probs >= 0.5).float()
                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_scores.extend(probs.cpu().numpy())

            elif outputs.ndim == 2 and outputs.shape[1] > 1:
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probs, 1)

                if probs.shape[1] == 2:
                    probs = probs[:, 1]

                y_true.extend(y_batch.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
                y_scores.extend(probs.cpu().numpy())

            else:
                raise ValueError(f"Nieoczekiwany kształt wyjścia modelu: {outputs.shape}")

    f1 = f1_score(y_true, y_pred, average='binary' if len(np.unique(y_true)) == 2 else 'macro')
    acc = accuracy_score(y_true, y_pred)

    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
        print(f"Final Accuracy: {acc:.4f} | F1 Score: {f1:.4f} | EER: {eer:.4f}")
    else:
        print(f"Final Accuracy: {acc:.4f} | F1 Score (macro): {f1:.4f}")