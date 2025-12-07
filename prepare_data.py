
from omegaconf import OmegaConf
import os
import numpy as np
import pandas as pd
import pywt
from ipywidgets import Audio
from scipy.fftpack import dct
from scipy.interpolate import interp1d
import soundfile as sf

import parselmouth
import librosa
from spafe.features.gfcc import gfcc
from spafe.features.lfcc import lfcc
from spafe.features.mfcc import mfcc

config = OmegaConf.load("config.yaml")

METADATA_PATH_LA = "LA-keys-full/keys/LA/CM/trial_metadata.txt"
FLAC_FOLDER_LA_1 = "ASVspoof2021_LA_eval/flac"
COLS_LA = ["speaker_id", "file_id", "codec", "corpus",
           "attack_id", "label", "trim", "set"]

InTheWild_df = "archive (1)/release_in_the_wild"

def prepare_filepaths(df, flac_folder, file_id_col="file_id"):

    df["file_name"] = df[file_id_col] + ".flac"
    df["filepath"] = df["file_name"].apply(lambda x: os.path.join(flac_folder, x))
    return df[df["filepath"].apply(os.path.exists)]


def detect_columns(metadata_path):
    preview = pd.read_csv(metadata_path, sep=r"\s+", header=None, nrows=5)
    n_cols = preview.shape[1]

    if n_cols == len(COLS_LA):
        return COLS_LA
    else:
        print(f"Niezgodna liczba kolumn ({n_cols}) w {metadata_path}, używam domyślnych nazw c0..c{n_cols - 1}")
        return [f"c{i}" for i in range(n_cols)]



def calculate_chunks(fpath, chunk_duration=2.0):
    try:
        info = sf.info(fpath)
        duration = info.frames / info.samplerate

        if duration < chunk_duration:
            print("Za krótkie:", fpath)
            return []

        full_chunks = int(duration // chunk_duration)
        chunks_info = []

        for i in range(full_chunks):
            chunks_info.append({
                "chunk_index": i,
                "chunk_start": i * chunk_duration,
                "chunk_end": (i + 1) * chunk_duration
            })

        return chunks_info

    except RuntimeError:
        print(f"OSTRZEŻENIE: nie można odczytać {fpath}")
        return []


def process_audio_rows(df, existing_paths=None):
    rows = []
    if existing_paths is None:
        existing_paths = set()

    for _, row in df.iterrows():
        fpath = row["filepath"]

        if fpath in existing_paths:
            continue

        chunks_data = calculate_chunks(fpath)

        for chunk in chunks_data:
            new_row = row.copy()
            new_row["chunk_index"] = chunk["chunk_index"]
            new_row["chunk_start"] = chunk["chunk_start"]
            new_row["chunk_end"] = chunk["chunk_end"]
            rows.append(new_row)

    return pd.DataFrame(rows)


def balance_class_distribution(df, min_per_class=200):
    if "label" not in df.columns:
        return df

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
        print(f"Za mało danych do balansowania (wymagane ≥{min_per_class} na klasę): {counts.to_dict()}")

    return df


def sample_dataset(df, sample_size):
    if sample_size and len(df) > sample_size:
        df = df.sample(min(len(df), sample_size)).copy()
        print(f"Zredukowano dane do {len(df)} próbek przez losowe próbkowanie.")
    return df


def split_train_test_data(final_df, sample_size):

    print(f"\nDzielę dane: {sample_size} do trenowania, reszta → TEST (zbalansowany).")

    final_df_train = final_df.sample(sample_size, random_state=42)
    remaining = final_df.drop(final_df_train.index)

    test_counts = remaining["label"].value_counts()
    min_test_class = test_counts.min()

    final_df_test = (
        remaining.groupby("label")
        .apply(lambda x: x.sample(min_test_class, random_state=42))
        .reset_index(drop=True)
    )

    print("Wielkość TEST:", len(final_df_test))
    print("Rozkład TEST:", final_df_test["label"].value_counts().to_dict())

    return final_df_train, final_df_test


def load_metadata_and_process(key, value, existing_paths):
    metadata_path = value['metadata']
    dfs_local = []

    for flac_folder in value['flac']:
        try:
            cols = detect_columns(metadata_path)
            df_raw = pd.read_csv(
                metadata_path, sep=r"\s+", header=None, names=cols, on_bad_lines='warn'
            )

            df_raw = prepare_filepaths(df_raw, flac_folder)
            if df_raw.empty:
                continue

            df_processed = process_audio_rows(df_raw, existing_paths)
            if df_processed.empty:
                continue

            print(f"Znaleziono {df_processed.shape[0]} fragmentów (2s) dla {key} w {os.path.basename(flac_folder)}")
            df_processed.to_csv(f"{key}_ratunkowe.csv")
            dfs_local.append(df_processed)

        except FileNotFoundError:
            print(f"OSTRZEŻENIE: Nie znaleziono pliku metadanych: {metadata_path}")

    return pd.concat(dfs_local, ignore_index=True) if dfs_local else pd.DataFrame()


def scan_directory_files(set_path, subset):
    result = []
    label_list = [l for l in os.listdir(set_path) if os.path.isdir(os.path.join(set_path, l))]

    for label in label_list:
        label_path = os.path.join(set_path, label)
        for file in os.listdir(label_path):
            result.append([subset, os.path.join(label_path, file), label])

    return pd.DataFrame(result, columns=['set', 'filepath', 'label'])


def prepare_dataframe(
        all_data,
        balance=True,
        sample_size=15000,
        min_per_class=400,
        df_train=None
):
    dfs = []
    existing_paths = set(
        df_train["filepath"].unique()) if df_train is not None and "filepath" in df_train.columns else set()

    for key, value in all_data.items():
        df = load_metadata_and_process(key, value, existing_paths)
        if df.empty:
            continue

        if balance and "label" in df.columns:
            df = balance_class_distribution(df, min_per_class)

            counts = df["label"].value_counts()
            if not (counts >= min_per_class).all():
                break

        if df_train is None:
            df = sample_dataset(df, sample_size)

        dfs.append(df)

    if not dfs:
        print("BŁĄD: Nie wczytano żadnych danych. Sprawdź ścieżki i konfigurację.")
        return pd.DataFrame(), pd.DataFrame()

    final_df = pd.concat(dfs, ignore_index=True, join="inner")

    print("\nŁącznie do przetworzenia:", len(final_df), "fragmentów po 2 sekundy.")
    if "label" in final_df.columns:
        print("Rozkład końcowy:", final_df["label"].value_counts().to_dict())

    if len(final_df) > sample_size and "label" in final_df.columns:
        return split_train_test_data(final_df, sample_size)
    else:
        print("\nZa mało danych na podział — zwracam tylko final_df.")
        return final_df, pd.DataFrame()


def prepare_dirs_dataset(dir_path, balance=True, min_per_class=None, sample_size=15000):
    dfs = []
    if min_per_class is None:
        min_per_class = {"train": 300, "val": 10, "test": 5}

    dir_list = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]

    for subset in dir_list:
        print(f"\nPrzetwarzanie katalogu: {subset}")
        set_path = os.path.join(dir_path, subset)

        df_wild = scan_directory_files(set_path, subset)
        df = process_audio_rows(df_wild)

        if df.empty:
            print(f"Brak danych w {subset}, pomijam.")
            continue

        df.to_csv(f"{subset}_ratunkowe.csv", index=False)
        print(f"Zapisano {subset}_ratunkowe.csv ({len(df)} rekordów)")

        if balance:
            min_required = min_per_class.get(subset, 5)
            df = balance_class_distribution(df, min_required)

        df = sample_dataset(df, sample_size)
        dfs.append(df)

    return dfs



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


def extract_mfcc(filepath, chunk_start=None, chunk_end=None, sr=None, n_mfcc=13, mean=False, augment=None):
    try:
        y, sr = librosa.load(filepath, sr=sr)

        if chunk_start is not None and chunk_end is not None:
            start_sample = int(chunk_start * sr)
            end_sample = min(int(chunk_end * sr), len(y))
            y = y[start_sample:end_sample]

        if augment is not None:
            y, sr = augment_audio(y, sr, mode=augment)

        mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        return np.mean(mfcc_feat, axis=1) if mean else mfcc_feat
    except Exception as e:
        print(f"[BŁĄD MFCC] {filepath}: {e}")
        return None


def extract_lfcc(filepath, chunk_start=None, chunk_end=None, n_ceps=13, mean=False, augment=None):
    try:
        y, sr = librosa.load(filepath, sr=None)
        if chunk_start is not None and chunk_end is not None:
            start_sample = int(chunk_start * sr)
            end_sample = min(int(chunk_end * sr), len(y))
            y = y[start_sample:end_sample]

        if augment is not None:
            y, sr = augment_audio(y, sr, mode=augment)

        y_int16 = (y * 32767).astype(np.int16)
        lfccs = lfcc(sig=y_int16, fs=sr, num_ceps=n_ceps)
        return np.mean(lfccs, axis=1) if mean else lfccs
    except Exception as e:
        print(f"[BŁĄD LFCC] {filepath}: {e}")
        return None


def extract_cqcc(filepath, chunk_start=None, chunk_end=None, sr=None,
                 bins_per_octave=12, n_ceps=19, mean=False, augment=None):
    try:
        y, sr = librosa.load(filepath, sr=sr)
        if chunk_start is not None and chunk_end is not None:
            start_sample = int(chunk_start * sr)
            end_sample = min(int(chunk_end * sr), len(y))
            y = y[start_sample:end_sample]

        if augment is not None:
            y, sr = augment_audio(y, sr, mode=augment)

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

        log_power = np.log(np.square(interp_cqt) + 1e-12)
        cqcc_coeffs = dct(log_power, type=2, axis=0, norm='ortho')[:n_ceps, :]

        if mean:
            cqcc_mean = np.mean(cqcc_coeffs, axis=1)
            return cqcc_mean

        return cqcc_coeffs

    except Exception as e:
        print(f"[BŁĄD CQCC] {filepath}: {e}")
        return None


def extract_gtcc(filepath, chunk_start=None, chunk_end=None, sr=None, n_filters=40, n_ceps=13, mean=False, augment=None):
    try:
        y, sr = librosa.load(filepath, sr=sr)
        if chunk_start is not None and chunk_end is not None:
            start_sample = int(chunk_start * sr)
            end_sample = min(int(chunk_end * sr), len(y))
            y = y[start_sample:end_sample]

        if augment is not None:
            y, sr = augment_audio(y, sr, mode=augment)

        gtccs = gfcc(sig=y, fs=sr, num_ceps=n_ceps, nfilts=n_filters)
        return np.mean(gtccs, axis=1) if mean else gtccs
    except Exception as e:
        print(f"[BŁĄD GTCC] {filepath}: {e}")
        return None


def extract_wpt(filepath, chunk_start=None, chunk_end=None, mean=False, augment=None):
    try:
        y, sr = librosa.load(filepath, sr=None)
        if chunk_start is not None and chunk_end is not None:
            start_sample = int(chunk_start * sr)
            end_sample = min(int(chunk_end * sr), len(y))
            y = y[start_sample:end_sample]

        if augment is not None:
            y, sr = augment_audio(y, sr, mode=augment)


        wp = pywt.WaveletPacket(data=y, wavelet='db4', mode='symmetric', maxlevel=3)
        wpt_feat = np.array([np.mean(np.square(node.data)) for node in wp.get_level(3, 'natural')])
        return wpt_feat
    except Exception as e:
        print(f"[BŁĄD WPT] {filepath}: {e}")
        return None


def extract_mel_spectrogram(filepath, chunk_start=None, chunk_end=None, sr=None, n_mels=64, fmax=None, mean=False, augment=None):
    try:
        y, sr = librosa.load(filepath, sr=sr)
        if chunk_start is not None and chunk_end is not None:
            start_sample = int(chunk_start * sr)
            end_sample = min(int(chunk_end * sr), len(y))
            y = y[start_sample:end_sample]

        if augment is not None:
            y, sr = augment_audio(y, sr, mode=augment)

        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=fmax or sr / 2)
        S_db = librosa.power_to_db(S, ref=np.max)
        return np.mean(S_db, axis=1) if mean else S_db
    except Exception as e:
        print(f"[BŁĄD MEL] {filepath}: {e}")
        return None

def augment_audio(data, sr, mode="change pitch", factor=None, listen_aug_audio=False):

    if mode == "change pitch":
        if factor is None:
            factor=1.025
        augmented = librosa.effects.pitch_shift(data, sr=sr, n_steps=factor)

    elif mode == "noise":
        if factor is None:
            factor=0.0125
        noise = np.random.randn(len(data))
        augmented = data + factor * noise
        augmented = augmented.astype(data.dtype)

    else:
        augmented = data

    if listen_aug_audio:
        return Audio(augmented, rate=sr)
    else:
        return augmented, sr