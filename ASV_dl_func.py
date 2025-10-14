
import os
import time
import numpy as np
import pandas as pd
import pywt
from scipy.fftpack import dct
from scipy.interpolate import interp1d
import soundfile as sf

import parselmouth
import librosa
from spafe.features.gfcc import gfcc
from spafe.features.lfcc import lfcc

from joblib import Parallel, delayed
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, FastICA
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split
)
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from train_fun import prepare_filepaths


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


def detect_columns(metadata_path):
    preview = pd.read_csv(metadata_path, sep=r"\s+", header=None, nrows=5)
    n_cols = preview.shape[1]

    if n_cols == len(COLS_DF):
        return COLS_DF
    elif n_cols == len(COLS_PA):
        return COLS_PA
    elif n_cols == len(COLS_LA):
        return COLS_LA
    else:
        print(f"Niezgodna liczba kolumn ({n_cols}) w {metadata_path}, używam domyślnych nazw c0..c{n_cols - 1}")
        return [f"c{i}" for i in range(n_cols)]


def prepare_dirs_dataset(dir_path, balance=True, min_per_class=None, sample_size=5000):
    dfs = []

    if min_per_class is None:
        min_per_class = {
            "train": 300,
            "val": 10,
            "test": 5,
        }

    dir_list = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]

    for subset in dir_list:
        result = []
        print(f"\nPrzetwarzanie katalogu: {subset}")

        set_path = os.path.join(dir_path, subset)
        label_list = [l for l in os.listdir(set_path) if os.path.isdir(os.path.join(set_path, l))]

        for label in label_list:
            label_path = os.path.join(set_path, label)
            for file in os.listdir(label_path):
                result.append([subset, os.path.join(label_path, file), label])

        df_wild = pd.DataFrame(result, columns=['set', 'filepath', 'label'])
        rows = []

        for _, row in df_wild.iterrows():
            fpath = row["filepath"]
            try:
                info = sf.info(fpath)
                duration = info.frames / info.samplerate

                if duration < 2.0:
                    print("Za krótkie:", fpath)
                    continue

                full_chunks = int(duration // 2.0)
                for i in range(full_chunks):
                    new_row = row.copy()
                    new_row["chunk_index"] = i
                    new_row["chunk_start"] = i * 2.0
                    new_row["chunk_end"] = (i + 1) * 2.0
                    rows.append(new_row)

            except RuntimeError:
                print(f"Nie można odczytać {fpath}")

        df = pd.DataFrame(rows)

        if df.empty:
            print(f"Brak danych w {subset}, pomijam.")
            continue

        df.to_csv(f"{subset}_ratunkowe.csv", index=False)
        print(f"Zapisano {subset}_ratunkowe.csv ({len(df)} rekordów)")

        if balance and "label" in df.columns:
            print("Rozkład etykiet przed balansowaniem:\n", df["label"].value_counts())
            counts = df["label"].value_counts()
            min_required = min_per_class.get(subset, 5)

            if (counts >= min_required).all():
                min_class = max(counts.min(), min_required)
                df = (
                    df.groupby("label")
                    .apply(lambda x: x.sample(min_class, random_state=42))
                    .reset_index(drop=True)
                )
                print(f"Zbalansowano klasy do {min_class} elementów każda.")
            else:
                print(f"Za mało danych do balansowania: {counts.to_dict()} (wymagane ≥{min_required})")

        if sample_size and len(df) > sample_size:
            df = df.sample(sample_size, random_state=42).copy()
            print(f"Próbkowanie: ograniczono do {sample_size} rekordów.")

        dfs.append(df)

    return dfs


def prepare_dataframe(
        all_data,
        balance=True,
        sample_size=2000,
        min_per_class=400,
        df_train=None
):
    dfs = []

    existing_paths = set(
        df_train["file_path"].unique()) if df_train is not None and "file_path" in df_train.columns else set()

    for key, value in all_data.items():
        metadata_path = value['metadata']
        for flac_folder in value['flac']:
            try:
                cols = detect_columns(metadata_path)
                df = pd.read_csv(
                    metadata_path, sep=r"\s+", header=None, names=cols, on_bad_lines='warn'
                )

                df = prepare_filepaths(df, flac_folder)
                if df.empty:
                    continue

                rows = []
                for _, row in df.iterrows():
                    fpath = row["file_path"]

                    if fpath in existing_paths:
                        continue

                    try:
                        info = sf.info(fpath)
                        duration = info.frames / info.samplerate

                        if duration < 2.0:
                            print("Za krótkie:", fpath)
                            continue

                        full_chunks = int(duration // 2.0)
                        for i in range(full_chunks):
                            new_row = row.copy()
                            new_row["chunk_index"] = i
                            new_row["chunk_start"] = i * 2.0
                            new_row["chunk_end"] = (i + 1) * 2.0
                            rows.append(new_row)

                    except RuntimeError:
                        print(f"OSTRZEŻENIE: nie można odczytać {fpath}")

                df = pd.DataFrame(rows)
                if df.empty:
                    continue

                print(f"Znaleziono {df.shape[0]} fragmentów (2s) dla {key} w {os.path.basename(flac_folder)}")
                df.to_csv(f"{key}_ratunkowe.csv")

                if balance and "label" in df.columns:
                    print("Rozkład etykiet:\n", df["label"].value_counts())
                    counts = df["label"].value_counts()
                    if (counts >= min_per_class).all():
                        min_class = max(min(counts), min_per_class)
                        df = (
                            df.groupby("label")
                            .apply(lambda x: x.sample(min_class, random_state=42))
                            .reset_index(drop=True)
                        )
                        print(f"Zbalansowano klasy do {min_class} elementów każda.")
                    else:
                        print(
                            f"Za mało danych do balansowania (wymagane ≥{min_per_class} na klasę): {counts.to_dict()}")
                        break

                if df_train is None and sample_size:
                    df = df.sample(min(len(df), sample_size)).copy()
                    print(f"Zredukowano dane do {len(df)} próbek przez losowe próbkowanie.")

                dfs.append(df)

            except FileNotFoundError:
                print(f"OSTRZEŻENIE: Nie znaleziono pliku metadanych: {metadata_path}")

    if not dfs:
        print("BŁĄD: Nie wczytano żadnych danych. Sprawdź ścieżki i konfigurację.")
        return pd.DataFrame()

    final_df = pd.concat(dfs, ignore_index=True, join="inner")

    print("\nŁącznie do przetworzenia:", len(final_df), "fragmentów po 2 sekundy.")
    if "label" in final_df.columns:
        print("Rozkład końcowy:", final_df["label"].value_counts().to_dict())

    return final_df


def analyze_formants_and_silence(filepath, silence_threshold_db=20, chunk_start=None, chunk_end=None, mean=False):
    try:
        snd = parselmouth.Sound(filepath)
        if chunk_start is not None and chunk_end is not None:
            snd = snd.extract_part(from_time=chunk_start, to_time=chunk_end)

        intensity = snd.to_intensity()
        intensity_values = intensity.values[0]
        silence_ratio = np.mean(intensity_values < silence_threshold_db)

        formant = snd.to_formant_burg()
        times = formant.ts()

        f1_values = np.array([formant.get_value_at_time(1, t) for t in times])
        f2_values = np.array([formant.get_value_at_time(2, t) for t in times])

        def get_segments(mask):
            segments, start = [], None
            for i, val in enumerate(mask):
                if val and start is None:
                    start = i
                elif not val and start is not None:
                    segments.append((start, i - 1))
                    start = None
            if start is not None:
                segments.append((start, len(mask) - 1))
            return segments

        def segments_durations(segments, times):
            return [times[end] - times[start] for start, end in segments if end > start]

        def safe_mean(arr):
            return np.mean(arr) if len(arr) > 0 else 0.0

        f1_segments = get_segments(~np.isnan(f1_values))
        f2_segments = get_segments(~np.isnan(f2_values))
        f1_durations = segments_durations(f1_segments, times)
        f2_durations = segments_durations(f2_segments, times)

        vtl_values = np.array([35000 / (4 * f1) if f1 > 0 else np.nan for f1 in f1_values])
        vtl_segments = get_segments(~np.isnan(vtl_values))
        vtl_durations = segments_durations(vtl_segments, times)

        return {
            "silence_ratio": silence_ratio,
            "f1_total_segments": len(f1_segments),
            "f2_total_segments": len(f2_segments),
            "f1_avg_duration": safe_mean(f1_durations),
            "f2_avg_duration": safe_mean(f2_durations),
            "f1_total_duration": np.sum(f1_durations),
            "f2_total_duration": np.sum(f2_durations),
            "vtl_total_segments": len(vtl_segments),
            "vtl_avg_duration": safe_mean(vtl_durations),
            "vtl_total_duration": np.sum(vtl_durations),
        }

    except Exception as e:
        print(f"[BŁĄD analyze_formants_and_silence] {filepath}: {e}")
        return None


def extract_mfcc(filepath, chunk_start=None, chunk_end=None, sr=None, n_mfcc=13, mean=False):
    try:
        y, sr = librosa.load(filepath, sr=sr)
        if chunk_start is not None and chunk_end is not None:
            start_sample = int(chunk_start * sr)
            end_sample = min(int(chunk_end * sr), len(y))
            y = y[start_sample:end_sample]

        mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc_feat, axis=1) if mean else mfcc_feat
    except Exception as e:
        print(f"[BŁĄD MFCC] {filepath}: {e}")
        return None


def extract_lfcc(filepath, chunk_start=None, chunk_end=None, n_ceps=13, mean=False):
    try:
        y, sr = librosa.load(filepath, sr=None)
        if chunk_start is not None and chunk_end is not None:
            start_sample = int(chunk_start * sr)
            end_sample = min(int(chunk_end * sr), len(y))
            y = y[start_sample:end_sample]

        y_int16 = (y * 32767).astype(np.int16)
        lfccs = lfcc(sig=y_int16, fs=sr, num_ceps=n_ceps)
        return np.mean(lfccs, axis=1) if mean else lfccs
    except Exception as e:
        print(f"[BŁĄD LFCC] {filepath}: {e}")
        return None


def extract_cqcc(filepath, chunk_start=None, chunk_end=None, sr=None,
                 bins_per_octave=12, n_ceps=19, mean=False):
    try:
        y, sr = librosa.load(filepath, sr=sr)
        if chunk_start is not None and chunk_end is not None:
            start_sample = int(chunk_start * sr)
            end_sample = min(int(chunk_end * sr), len(y))
            y = y[start_sample:end_sample]

        fmin = librosa.note_to_hz('C1')
        fmax = sr / 2 - 100
        n_bins = int(np.floor(np.log2(fmax / fmin)) * bins_per_octave)

        cqt = librosa.cqt(y, sr=sr, n_bins=n_bins, bins_per_octave=bins_per_octave, fmin=fmin)
        cqt_mag = np.abs(cqt)
        cqt_db = librosa.amplitude_to_db(cqt_mag, ref=np.max)

        original_freqs = librosa.cqt_frequencies(n_bins=n_bins, fmin=fmin, bins_per_octave=bins_per_octave)
        lin_freqs = np.linspace(original_freqs.min(), original_freqs.max(), num=n_bins)

        interp_cqt = np.zeros_like(cqt_db)
        for t in range(cqt_db.shape[1]):
            interp_func = interp1d(original_freqs, cqt_db[:, t], kind='linear', fill_value="extrapolate")
            interp_cqt[:, t] = interp_func(lin_freqs)

        interp_cqt = np.maximum(interp_cqt, 1e-8)
        interp_lin = np.power(10.0, interp_cqt / 20.0)
        log_power = np.log1p(interp_lin)

        cqcc_coeffs = dct(log_power, type=2, axis=0, norm='ortho')[:n_ceps, :]

        if mean:
            cqcc_mean = np.mean(cqcc_coeffs, axis=1)
            return cqcc_mean

        return cqcc_coeffs

    except Exception as e:
        print(f"[BŁĄD CQCC] {filepath}: {e}")
        return None


def extract_gtcc(filepath, chunk_start=None, chunk_end=None, sr=None, n_filters=40, n_ceps=13, mean=False):
    try:
        y, sr = librosa.load(filepath, sr=sr)
        if chunk_start is not None and chunk_end is not None:
            start_sample = int(chunk_start * sr)
            end_sample = min(int(chunk_end * sr), len(y))
            y = y[start_sample:end_sample]

        gtccs = gfcc(sig=y, fs=sr, num_ceps=n_ceps, nfilts=n_filters)
        return np.mean(gtccs, axis=1) if mean else gtccs
    except Exception as e:
        print(f"[BŁĄD GTCC] {filepath}: {e}")
        return None


def extract_wpt(filepath, chunk_start=None, chunk_end=None, mean=False):
    try:
        y, sr = librosa.load(filepath, sr=None)
        if chunk_start is not None and chunk_end is not None:
            start_sample = int(chunk_start * sr)
            end_sample = min(int(chunk_end * sr), len(y))
            y = y[start_sample:end_sample]

        wp = pywt.WaveletPacket(data=y, wavelet='db4', mode='symmetric', maxlevel=3)
        wpt_feat = np.array([np.mean(np.square(node.data)) for node in wp.get_level(3, 'natural')])
        return wpt_feat
    except Exception as e:
        print(f"[BŁĄD WPT] {filepath}: {e}")
        return None


def extract_mel_spectrogram(filepath, chunk_start=None, chunk_end=None, sr=None, n_mels=64, fmax=None, mean=False):
    try:
        y, sr = librosa.load(filepath, sr=sr)
        if chunk_start is not None and chunk_end is not None:
            start_sample = int(chunk_start * sr)
            end_sample = min(int(chunk_end * sr), len(y))
            y = y[start_sample:end_sample]

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax or sr / 2)
        S_db = librosa.power_to_db(S, ref=np.max)
        return np.mean(S_db, axis=1) if mean else S_db
    except Exception as e:
        print(f"[BŁĄD MEL] {filepath}: {e}")
        return None


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.stride = stride

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.LeakyReLU(negative_slope=0.01)
        self.dropout = nn.Dropout2d(p=0.5)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout(out)

        out = self.conv2(out)

        identity = self.shortcut(identity)

        out += identity
        out = self.bn2(out)
        out = self.relu2(out)
        return out


class AntiSpoofingResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(AntiSpoofingResNet, self).__init__()

        self.initial_sequence = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.01)
        )
        self.residual_blocks = nn.Sequential(
            ResidualBlock(32, 32, stride=3),
            ResidualBlock(32, 32, stride=1),
            ResidualBlock(32, 32, stride=1),
            ResidualBlock(32, 32, stride=1),
            ResidualBlock(32, 32, stride=1),
            ResidualBlock(32, 32, stride=1)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(32, 256),
            nn.Dropout(p=0.5),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        out = self.initial_sequence(x)
        out = self.residual_blocks(out)
        out = self.avg_pool(out)
        out = torch.flatten(out, 1)

        out = self.classifier(out)

        return out


class FeatureColumnDataset(Dataset):
    def __init__(self, df, feature_col, label_col='label'):
        self.features = df[feature_col].values
        self.labels = df[label_col].values

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = np.array(self.features[idx])
        y = self.labels[idx]
        if x.ndim == 1:
            x = x[np.newaxis, :, np.newaxis]
        elif x.ndim == 2:
            x = x[np.newaxis, :, :]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)


def train_feature_model(final_df, feature_col, label_col='label', batch_size=32, epochs=10, device=None, test_df=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if test_df is None:
        X_train_df, X_test_df = train_test_split(final_df, test_size=0.2, stratify=final_df[label_col], random_state=42)
    else:
        X_train_df = final_df
        X_test_df = test_df

    train_dataset = FeatureColumnDataset(X_train_df, feature_col, label_col)
    test_dataset = FeatureColumnDataset(X_test_df, feature_col, label_col)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = AntiSpoofingResNet(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

        train_loss = running_loss / total
        train_acc = correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item() * X_batch.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += y_batch.size(0)
                val_correct += (predicted == y_batch).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"[{feature_col}] Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    return model, test_loader


def train_all_features(final_df, feature_cols, test_df=None, label_col='label', batch_size=32, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_models = {}

    for feat in feature_cols:
        print(f"\n=== TRENING dla cechy: {feat} ===")

        df_clean = final_df.dropna(subset=[feat, label_col]).reset_index(drop=True)

        if df_clean.empty:
            print(f"[UWAGA] Brak danych do treningu dla cechy {feat} po usunięciu NaN!")
            continue

        model, test_loader = train_feature_model(df_clean, feat, label_col, batch_size, epochs, device, test_df=test_df)
        trained_models[feat] = [model, test_loader]
        print(f"=== KONIEC treningu dla cechy: {feat} ===\n")

    return trained_models


def extract_features(final_df, feature_extractors_map, col_name='filepath', mean=False):
    print("mean ", mean)
    for name, func in feature_extractors_map.items():
        print(f"   - Ekstrahuję: {name}")

        results = Parallel(n_jobs=-1, verbose=1)(
            delayed(func)(
                row[col_name],
                chunk_start=row.get('chunk_start', None),
                chunk_end=row.get('chunk_end', None),
                mean=mean
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


def prepare_train_test_data(df, test_df=None, col_name="cqcc", label_name="label_num"):
    if test_df is None:
        train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df[label_name])
    else:
        train_df = df

    scaler = StandardScaler()

    all_train_features_for_scaler = np.vstack(train_df[col_name].values)
    scaler.fit(all_train_features_for_scaler)

    train_df[col_name] = train_df[col_name].apply(lambda x: scaler.transform(x))
    test_df[col_name] = test_df[col_name].apply(lambda x: scaler.transform(x))

    return train_df, test_df, scaler


def gmm_model(train_df, N_COMPONENTS_GMM=128, feature_name='cqcc', label_name="label_num"):
    print("Trening Gaussian Mixture (UBM)...")

    all_train_features_gmm = np.vstack(train_df['cqcc'].values)
    ubm = GaussianMixture(n_components=N_COMPONENTS_GMM, covariance_type='diag', max_iter=100, random_state=42,
                          verbose=1)
    start_time_ubm = time.time()
    ubm.fit(all_train_features_gmm)
    end_time_ubm = time.time()
    print(f"Trening UBM zakończony w {end_time_ubm - start_time_ubm:.2f} sekund.")

    print("Adaptacja GMM dla klas Genuine i DF...")
    start_time_map = time.time()
    gmm_genuine = map_adapt(ubm, np.vstack(train_df[train_df[label_name] == 0][feature_name].values))
    gmm_df = map_adapt(ubm, np.vstack(train_df[train_df[label_name] == 1][feature_name].values))
    end_time_map = time.time()
    print(f"Adaptacja MAP zakończona w {end_time_map - start_time_map:.2f} sekund.")

    return gmm_genuine, gmm_df


def map_adapt(gmm_ubm, features, relevance_factor=10, max_iterations=20):
    gmm_class = GaussianMixture(n_components=gmm_ubm.n_components, covariance_type='diag', random_state=42)
    gmm_class.weights_ = np.copy(gmm_ubm.weights_)
    gmm_class.means_ = np.copy(gmm_ubm.means_)
    gmm_class.covariances_ = np.copy(gmm_ubm.covariances_)

    for _ in range(max_iterations):
        responsibilities = gmm_ubm.predict_proba(features)
        N_k = responsibilities.sum(axis=0) + 1e-6
        F_k = np.dot(responsibilities.T, features)
        alpha_mean = N_k / (N_k + relevance_factor)
        new_means = (alpha_mean[:, np.newaxis] * (F_k / N_k[:, np.newaxis])) + (
                    (1 - alpha_mean[:, np.newaxis]) * gmm_ubm.means_)
        gmm_class.means_ = new_means

        S_k = np.dot(responsibilities.T, features ** 2)
        new_vars = (alpha_mean[:, np.newaxis] * (S_k / N_k[:, np.newaxis] - new_means ** 2)) + (
                    (1 - alpha_mean[:, np.newaxis]) * gmm_ubm.covariances_)
        gmm_class.covariances_ = np.maximum(new_vars, 1e-6)

        alpha_weight = N_k / (N_k + relevance_factor)
        new_weights = (alpha_weight * (N_k / N_k.sum())) + ((1 - alpha_weight) * gmm_ubm.weights_)
        gmm_class.weights_ = new_weights / new_weights.sum()

    gmm_class.precisions_cholesky_ = 1.0 / np.sqrt(gmm_class.covariances_)
    return gmm_class


def compute_llr(features, gmm1, gmm2):
    ll1 = gmm1.score(features)
    ll2 = gmm2.score(features)
    return ll1 - ll2


class AudioDataset(Dataset):
    def __init__(self, df, label_name='label'):
        self.features = df['cqcc'].values
        self.labels = df[label_name].values.astype(np.int64)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


def collate_fn_padd(batch):
    features, labels = zip(*batch)

    padded_features = pad_sequence(features, batch_first=True, padding_value=0.0)

    labels = torch.tensor(labels, dtype=torch.long)

    return padded_features, labels


def BiLSTM_model(train_df, test_df, col_name="cqcc", num_epochs=100, model=None, criterion=None, optimizer=None):
    print("Tworzenie DataLoaderów...")
    train_dataset = AudioDataset(train_df)
    test_dataset = AudioDataset(test_df)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=False, collate_fn=collate_fn_padd)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=False, collate_fn=collate_fn_padd)
    print(f"Liczba partii treningowych: {len(train_loader)}, Liczba partii testowych: {len(test_loader)}")

    if len(train_df[col_name]) == 0:
        raise ValueError("Brak danych treningowych po przygotowaniu i normalizacji.")

    if train_df[col_name].iloc[0].shape[0] == 0:
        for features_array in train_df[col_name].values:
            if features_array.shape[0] > 0:
                input_dim = features_array.shape[1]
                break
        else:
            raise ValueError(f"Wszystkie sekwencje {col_name} w train_df są puste po przygotowaniu.")
    else:
        input_dim = train_df[col_name].iloc[0].shape[1]

    if model is None:
        model = BiLSTMClassifier(input_dim=input_dim)
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print("Początek pętli treningowej BiLSTM...")
    start_time_bilstm = time.time()

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            if X_batch.size(0) == 0:
                print(f"Ostrzeżenie: Pusta partia treningowa w epoce {epoch + 1}, batch {batch_idx}. Pomijam.")
                continue

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X_val, y_val in test_loader:
                if X_val.size(0) == 0:
                    continue

                X_val = X_val.to(device)
                y_val = y_val.to(device)
                outputs = model(X_val)
                loss = criterion(outputs, y_val)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += y_val.size(0)
                correct += (predicted == y_val).sum().item()

        avg_val_loss = val_loss / len(test_loader)
        val_accuracy = correct / total if total > 0 else 0

        print(f"Epoch {epoch + 1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.4f}")

    end_time_bilstm = time.time()
    print(f"Trening BiLSTM zakończony w {end_time_bilstm - start_time_bilstm:.2f} sekund.")

    return model, test_loader


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=1, dropout=0.3):
        super().__init__()
        self.bi_lstm1 = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                                batch_first=True, bidirectional=True)
        self.dropout1 = nn.Dropout(dropout)
        self.bi_lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers=num_layers,
                                batch_first=True, bidirectional=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, x):
        out, _ = self.bi_lstm1(x)
        out = self.dropout1(out)
        out, _ = self.bi_lstm2(out)
        out = self.dropout2(out)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def fused_score(model, x_tensor, features_np, gmm_genuine, gmm_df):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    with torch.no_grad():
        # BiLSTM
        x_tensor_gpu = x_tensor.unsqueeze(0).to(device)
        bi_lstm_output = model(x_tensor_gpu)
        bi_lstm_prob = torch.softmax(bi_lstm_output, dim=1).cpu().numpy().squeeze()[1]

        # GMM
        gmm_llr = compute_llr(features_np, gmm_genuine, gmm_df)
        gmm_prob = 1 / (1 + np.exp(-gmm_llr))

        return 0.5 * bi_lstm_prob + 0.5 * gmm_prob


def eval_model(model, train_df, test_df, test_loader, feature_name: str = 'cqcc',
               label_name: str = "label", use_melspect = False):
    print("Rozpoczynanie ewaluacji...")

    gmm_genuine, gmm_df = gmm_model(train_df, feature_name=feature_name, label_name=label_name)

    y_true = []
    y_pred = []

    start_time_eval = time.time()
    if len(test_loader) == 0 and len(test_df) > 0:
        print("Ostrzeżenie: test_loader jest pusty, ale test_df ma dane. Może być problem z batch_size lub collate_fn.")

    for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
        if X_batch.size(0) == 0:
            print(f"Ostrzeżenie: Pusta partia testowa w batch {batch_idx}. Pomijam.")
            continue

        for i in range(X_batch.size(0)):
            sample_x_tensor = X_batch[i]

            original_features_mask = (sample_x_tensor.sum(dim=1) != 0)
            print("original_features_mask: ", original_features_mask)
            print("sample_x_tensor: ", sample_x_tensor)
            sample_features_np = sample_x_tensor[original_features_mask].cpu().numpy()

            if sample_features_np.shape[0] == 0:
                score = 0.5
                print(f"Ostrzeżenie: Próbka {i} w partii {batch_idx} jest cała z paddingu. Przypisuję score 0.5.")
            else:
                score = fused_score(model, sample_x_tensor, sample_features_np, gmm_genuine, gmm_df)

            y_pred.append(1 if score > 0.5 else 0)

        y_true.extend(y_batch.numpy())

    end_time_eval = time.time()
    print(f"Ewaluacja zakończona w {end_time_eval - start_time_eval:.2f} sekund.")

    print("\n--- Wyniki końcowe ---")
    if len(y_true) == 0:
        print("Brak danych do ewaluacji (y_true jest puste).")
    else:
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("F1:", f1_score(y_true, y_pred))


def _to_array_safe(x):

    if x is None:
        return np.array([])
    if isinstance(x, np.ndarray):
        arr = x
    else:
        try:
            arr = np.array(x)
        except Exception:
            arr = np.array([x])

    arr = np.ravel(arr)
    return arr


def build_X_from_df(df, feature_cols):

    rows = []
    for _, row in df.iterrows():
        parts = []
        for c in feature_cols:
            cell = row[c]
            arr = _to_array_safe(cell)
            parts.append(arr)
        if parts:
            flat = np.hstack([p for p in parts if p.size > 0])
        else:
            flat = np.array([])
        rows.append(flat)

    lengths = [r.size for r in rows]
    if len(set(lengths)) != 1:
        max_len = max(lengths)
        padded = np.zeros((len(rows), max_len), dtype=float)
        for i, r in enumerate(rows):
            padded[i, : r.size] = r
        X = padded
    else:
        X = np.vstack(rows) if rows else np.empty((0, 0))
    return X

def generate_feature_sets(feature_cols):

    yield ('all', list(feature_cols))

    for c in feature_cols:
        yield (c, [c])

    for c in feature_cols:
        for other in feature_cols:
            if other == c:
                continue
            yield (f"{c}_plus_{other}", [c, other])


def make_pipeline(reducer_name=None, n_components=None, standardize=False, classifier=None):
    steps = []
    if standardize:
        steps.append(('scaler', StandardScaler()))

    if reducer_name == 'pca':
        steps.append(('reducer', PCA(n_components=n_components, random_state=101)))
    elif reducer_name == 'ica':
        steps.append(('reducer', FastICA(n_components=n_components, random_state=101)))


    steps.append(('clf', classifier))
    return Pipeline(steps)

def run_extensive_gridsearch(
    df_train,
    df_test,
    feature_cols=None,
    svm_params=None,
    xgb_params=None,
    cv=3,
    scoring='accuracy',
    n_jobs=-1,
    min_samples=10,
    top_k=5,
    label_col='label',
    out_dir='gridsearch_results',
    random_state=42
):

    if svm_params is None:
        svm_params = {
            'clf__C': [1, 10],
            'clf__kernel': ['rbf'],
            'clf__gamma': ['scale', 0.1]
        }

    if xgb_params is None:
        xgb_params = {
            'clf__n_estimators': [50, 100],
            'clf__max_depth': [3, 5],
            'clf__learning_rate': [0.1, 0.05],
            'clf__subsample': [0.8, 1],
            'clf__colsample_bytree': [0.8, 1],
            'clf__gamma': [0, 1]
        }

    pca_components = [10, 20, 30, 40]
    ica_components = [10, 20, 30, 40]
    os.makedirs(out_dir, exist_ok=True)

    y_train = df_train[label_col].values
    y_test = df_test[label_col].values

    def build_X_from_df(df, cols):
        return df[cols].values

    def generate_feature_sets(feature_cols):
        if feature_cols is None:
            yield ("all_features", [col for col in df_train.columns if col != label_col])
        else:
            yield ("selected_features", feature_cols)

    def make_pipeline(reducer, n_comp, standardize, classifier):
        steps = []
        if standardize:
            steps.append(('scaler', StandardScaler()))
        if reducer == 'pca' and n_comp is not None:
            steps.append(('reducer', PCA(n_components=n_comp, random_state=random_state)))
        elif reducer == 'ica' and n_comp is not None:
            steps.append(('reducer', FastICA(n_components=n_comp, random_state=random_state)))
        steps.append(('clf', classifier))
        return Pipeline(steps)

    results_summary = []

    for fs_name, fs_cols in generate_feature_sets(feature_cols):
        print(f"\n== Feature set: {fs_name} -> {fs_cols}")
        X_train = build_X_from_df(df_train, fs_cols)
        X_test = build_X_from_df(df_test, fs_cols)

        if X_train.shape[0] < min_samples:
            print(f"  Pomijam zestaw '{fs_name}' — za mało próbek ({X_train.shape[0]})")
            continue

        n_features = X_train.shape[1]
        print(f"  {fs_name}: {n_features} cech po rozpakowaniu.")

        for standardize in [False, True]:
            for reducer in [None, 'pca', 'ica']:
                comp_list = [None]
                if reducer == 'pca':
                    comp_list = pca_components
                elif reducer == 'ica':
                    comp_list = ica_components

                for n_comp in comp_list:
                    if n_comp is not None and n_comp >= n_features:
                        continue

                    svm = SVC(probability=True, random_state=random_state)
                    pipe_svm = make_pipeline(reducer, n_comp, standardize, svm)
                    print(f"  SVM | std={standardize} | reducer={reducer} | n_comp={n_comp}")
                    try:
                        gs_svm = GridSearchCV(
                            pipe_svm,
                            svm_params,
                            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
                            scoring=scoring,
                            n_jobs=n_jobs,
                            verbose=0
                        )
                        gs_svm.fit(X_train, y_train)
                        y_pred = gs_svm.predict(X_test)
                        test_acc = accuracy_score(y_test, y_pred)
                        class_rep = classification_report(y_test, y_pred, output_dict=True)
                    except Exception as e:
                        print(f"    Błąd SVM: {e}")
                        continue

                    results_summary.append({
                        'model': 'SVM',
                        'feature_set': fs_name,
                        'std': standardize,
                        'reducer': reducer,
                        'n_comp': n_comp,
                        'train_score': gs_svm.best_score_,
                        'test_score': test_acc,
                        'precision': class_rep['weighted avg']['precision'],
                        'recall': class_rep['weighted avg']['recall'],
                        'f1': class_rep['weighted avg']['f1-score'],
                        'best_params': gs_svm.best_params_
                    })


                    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
                    pipe_xgb = make_pipeline(reducer, n_comp, standardize, xgb)
                    print(f"  XGB | std={standardize} | reducer={reducer} | n_comp={n_comp}")
                    try:
                        gs_xgb = GridSearchCV(
                            pipe_xgb,
                            xgb_params,
                            cv=StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state),
                            scoring=scoring,
                            n_jobs=n_jobs,
                            verbose=0
                        )
                        gs_xgb.fit(X_train, y_train)
                        y_pred_xgb = gs_xgb.predict(X_test)
                        test_acc_xgb = accuracy_score(y_test, y_pred_xgb)
                        class_rep_xgb = classification_report(y_test, y_pred_xgb, output_dict=True)
                    except Exception as e:
                        print(f"    Błąd XGB: {e}")
                        continue

                    results_summary.append({
                        'model': 'XGB',
                        'feature_set': fs_name,
                        'std': standardize,
                        'reducer': reducer,
                        'n_comp': n_comp,
                        'train_score': gs_xgb.best_score_,
                        'test_score': test_acc_xgb,
                        'precision': class_rep_xgb['weighted avg']['precision'],
                        'recall': class_rep_xgb['weighted avg']['recall'],
                        'f1': class_rep_xgb['weighted avg']['f1-score'],
                        'best_params': gs_xgb.best_params_
                    })

    df_res = pd.DataFrame(results_summary)
    if df_res.empty:
        print("Brak wyników do zapisania.")
        return None, None

    df_res = df_res.sort_values(by="test_score", ascending=False).reset_index(drop=True)
    top_models = df_res.head(top_k)

    df_res.to_csv(os.path.join(out_dir, 'gridsearch_summary.csv'), index=False)
    top_models.to_csv(os.path.join(out_dir, f'top_{top_k}_models.csv'), index=False)

    print("\nNajlepsze modele na danych testowych:")
    print(top_models[['model', 'feature_set', 'test_score', 'precision', 'recall', 'f1', 'best_params']])

    return top_models, df_res
