from IPython.display import Audio, display
import pandas as pd
import pywt
from spafe.features.lfcc import lfcc
from matplotlib import pyplot as plt
from spafe.features.gfcc import gfcc
import numpy as np
import librosa
from scipy.fftpack import dct
from scipy.interpolate import interp1d
import os
from omegaconf import OmegaConf

OmegaConf.register_new_resolver(
    "env",
    lambda var, default=None: os.environ.get(var, default)
)

config = OmegaConf.load("config.yaml")

meta = config.paths.metadata_path
flac = config.paths.flac_folder

print("Metadata path:", meta)
print("FLAC folder:", flac)


def prepare_filepath(df, file_id_col="file_id"):
    df["file_name"] = df[file_id_col] + ".flac"
    df["file_path"] = df["file_name"].apply(lambda x: os.path.join(flac, x))

    return df[df["file_path"].apply(os.path.exists)]


def listen_voice_flac(df, n_samples=5, file_path="file_path", label="label"):
    samples = df.sample(n_samples, random_state=42)[[file_path, label]].reset_index(drop=True)
    for i, row in samples.iterrows():
        print(f"{i + 1}. {row[label].upper()} — {os.path.basename(row[file_path])}")
        y, sr = librosa.load(row[file_path], sr=None)
        display(Audio(y, rate=sr))


def extract_mfcc(filepath, chunk_start=None, chunk_end=None, sr=None, n_mfcc=13, mean=True):

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


def extract_lfcc(filepath, chunk_start=None, chunk_end=None, n_ceps=13, mean=True):

    try:
        y, sr = librosa.load(filepath, sr=None)
        if chunk_start is not None and chunk_end is not None:
            start_sample = int(chunk_start * sr)
            end_sample = min(int(chunk_end * sr), len(y))
            y = y[start_sample:end_sample]

        y_int16 = (y * 32767).astype(np.int16)
        lfccs = lfcc(sig=y_int16, fs=sr, num_ceps=n_ceps)
        return np.mean(lfccs, axis=0) if mean else lfccs
    except Exception as e:
        print(f"[BŁĄD LFCC] {filepath}: {e}")
        return None


def extract_cqcc(filepath, chunk_start=None, chunk_end=None, sr=None, bins_per_octave=12, n_ceps=19, mean=True):

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

        interp_cqt = np.zeros((n_bins, cqt_db.shape[1]))
        for t in range(cqt_db.shape[1]):
            interp_func = interp1d(original_freqs, cqt_db[:, t], kind='cubic', fill_value="extrapolate")
            interp_cqt[:, t] = interp_func(lin_freqs)

        log_power = np.log(np.square(interp_cqt) + 1e-12)
        cqcc_coeffs = dct(log_power, type=2, axis=0, norm='ortho')[:n_ceps, :]
        return np.mean(cqcc_coeffs, axis=1) if mean else cqcc_coeffs
    except Exception as e:
        print(f"[BŁĄD CQCC] {filepath}: {e}")
        return None


def extract_gtcc(filepath, chunk_start=None, chunk_end=None, sr=None, n_filters=40, n_ceps=13, mean=True):

    try:
        y, sr = librosa.load(filepath, sr=sr)
        if chunk_start is not None and chunk_end is not None:
            start_sample = int(chunk_start * sr)
            end_sample = min(int(chunk_end * sr), len(y))
            y = y[start_sample:end_sample]

        gtccs = gfcc(sig=y, fs=sr, num_ceps=n_ceps, nfilts=n_filters)
        return np.mean(gtccs, axis=0) if mean else gtccs
    except Exception as e:
        print(f"[BŁĄD GTCC] {filepath}: {e}")
        return None


def extract_wpt(filepath, chunk_start=None, chunk_end=None, mean=True):

    try:
        y, sr = librosa.load(filepath, sr=None)
        if chunk_start is not None and chunk_end is not None:
            start_sample = int(chunk_start * sr)
            end_sample = min(int(chunk_end * sr), len(y))
            y = y[start_sample:end_sample]

        wp = pywt.WaveletPacket(data=y, wavelet='db4', mode='symmetric', maxlevel=3)
        wpt_feat = np.array([np.mean(np.square(node.data)) for node in wp.get_level(3, 'natural')])
        return np.mean(wpt_feat) if mean else wpt_feat
    except Exception as e:
        print(f"[BŁĄD WPT] {filepath}: {e}")
        return None


def extract_mel_spectrogram(filepath, chunk_start=None, chunk_end=None, sr=None, n_mels=64, fmax=None, mean=True):

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


def plot_coeff_histograms_by_label_separately(df, coeff_col='MFCC', label_col='label'):
    n_coeffs = len(df[coeff_col].iloc[0])

    mfcc_df = pd.DataFrame(df[coeff_col].tolist(), columns=[f'{coeff_col}_{i + 1}' for i in range(n_coeffs)])

    df_full = pd.concat([
        df[label_col].reset_index(drop=True),
        mfcc_df.reset_index(drop=True)
    ], axis=1)

    labels = df_full[label_col].unique()
    default_colors = ['skyblue', 'salmon', 'lightgreen', 'plum', 'orange', 'gold', 'lightcoral']
    label_colors = dict(zip(labels, default_colors[:len(labels)]))

    for i in range(n_coeffs):
        col_name = f'{coeff_col}_{i + 1}'

        plt.figure(figsize=(6, 4))

        for label in labels:
            subset = df_full[df_full[label_col] == label][col_name]
            if not subset.dropna().empty:
                plt.hist(subset, bins=10, alpha=0.6, label=label,
                         color=label_colors[label], edgecolor='black')

        plt.title(col_name)
        plt.xlabel('Wartość')
        plt.ylabel('Liczba wystąpień')
        plt.legend()
        plt.tight_layout()
        plt.show()
