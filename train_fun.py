
import os
import pandas as pd
import numpy as np
import librosa
import soundfile as sf
import pywt
import parselmouth
from scipy.fftpack import dct
from scipy.interpolate import interp1d
from spafe.features.lfcc import lfcc
from spafe.features.gfcc import gfcc
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from xgboost import XGBClassifier
from joblib import Memory, Parallel, delayed
from IPython.display import Audio, display
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import warnings

warnings.filterwarnings('ignore', category=UserWarning)

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




def prepare_filepaths(df, flac_folder, file_id_col="file_id"):

    df["file_name"] = df[file_id_col] + ".flac"
    df["file_path"] = df["file_name"].apply(lambda x: os.path.join(flac_folder, x))
    return df[df["file_path"].apply(os.path.exists)]


def listen_voice_flac(df, file_path_col="file_path", label_col="label"):

    samples = df.sample(5, random_state=42)[[file_path_col, label_col]].reset_index(drop=True)
    for i, row in samples.iterrows():
        print(f"{i + 1}. {row[label_col].upper()} — {os.path.basename(row[file_path_col])}")
        try:
            y, sr = librosa.load(row[file_path_col], sr=None)
            display(Audio(y, rate=sr))
        except Exception as e:
            print(f"  Nie można wczytać pliku: {e}")



def extract_mfcc(filepath):

    try:
        y, sr = librosa.load(filepath, sr=None)
        mfcc_feat = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1)
        return mfcc_feat
    except Exception:
        print(Exception)
        return None


def extract_lfcc(filepath):

    try:
        y, sr = librosa.load(filepath, sr=None)
        y_int16 = (y * 32767).astype(np.int16)
        lfccs = lfcc(sig=y_int16, fs=sr, num_ceps=13)
        return np.mean(lfccs, axis=0)
    except Exception:
        return None


def extract_cqcc(filepath, sr=None, bins_per_octave=12, n_ceps=19):

    try:
        y, sr = librosa.load(filepath, sr=sr)
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
        cqcc_coeffs = dct(log_power, type=2, axis=0, norm='ortho')
        cqcc_coeffs = cqcc_coeffs[:n_ceps, :]

        return np.mean(cqcc_coeffs, axis=1)
    except Exception:
        print(Exception)
        return None


def extract_gtcc(filepath, sr=None, n_filters=40, n_ceps=13):

    try:
        y, sr = librosa.load(filepath, sr=sr)
        gtccs = gfcc(sig=y, fs=sr, num_ceps=n_ceps, nfilts=n_filters)
        return np.mean(gtccs, axis=0)
    except Exception:
        print(Exception)
        return None


def extract_wpt(filepath):

    try:
        y, sr = librosa.load(filepath, sr=None)
        wp = pywt.WaveletPacket(data=y, wavelet='db4', mode='symmetric', maxlevel=3)
        wpt_feat = np.array([np.mean(np.square(node.data)) for node in wp.get_level(3, 'natural')])
        return wpt_feat
    except Exception:
        print(Exception)
        return None


def analyze_formants_and_silence(filepath, silence_threshold_db=20):

    try:
        snd = parselmouth.Sound(filepath)
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
                    segments.append((start, i - 1)); start = None
            if start is not None: segments.append((start, len(mask) - 1))
            return segments

        def segments_durations(segments, times):
            return [times[end] - times[start] for start, end in segments]

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
            "f1_total_segments": len(f1_segments), "f2_total_segments": len(f2_segments),
            "f1_avg_duration": safe_mean(f1_durations), "f2_avg_duration": safe_mean(f2_durations),
            "f1_total_duration": np.sum(f1_durations), "f2_total_duration": np.sum(f2_durations),
            "vtl_total_segments": len(vtl_segments), "vtl_avg_duration": safe_mean(vtl_durations),
            "vtl_total_duration": np.sum(vtl_durations),
        }
    except Exception:
        print(Exception)
        return None



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



def expand_feature_columns(df, feature_columns, label_column):

    expanded_features_list = []
    for col in feature_columns:

        df_clean = df.dropna(subset=[col])


        if isinstance(df_clean[col].iloc[0], dict):
            expanded = df_clean[col].apply(pd.Series)
            expanded.columns = [f"{col}_{key}" for key in expanded.columns]
        else:
            expanded = pd.DataFrame(df_clean[col].tolist())
            expanded.columns = [f"{col}_{i}" for i in range(expanded.shape[1])]

        expanded_features_list.append(expanded)


    final_expanded_df = pd.concat(expanded_features_list, axis=1)
    return pd.concat([final_expanded_df, df[label_column]], axis=1).dropna()


def preprocess_for_modeling(df, label_col='label'):

    df_copy = df.copy()
    if label_col not in df_copy.columns:
        raise ValueError(f"Kolumna etykiet '{label_col}' nie została znaleziona w ramce danych.")

    df_copy[label_col] = df_copy[label_col].astype(str).str.lower().map({'spoof': 1, 'bonafide': 0})
    df_copy = df_copy.dropna(subset=[label_col])


    X = df_copy.drop(columns=[label_col])
    y = df_copy[label_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    train_df = pd.concat([X_train, y_train], axis=1)
    class_0 = train_df[train_df[label_col] == 0]
    class_1 = train_df[train_df[label_col] == 1]

    if len(class_1) == 0 or len(class_0) == 0:
        raise ValueError("Zbiór treningowy nie zawiera próbek z obu klas. Nie można przeprowadzić oversamplingu.")


    if len(class_0) > len(class_1):
        class_1_upsampled = resample(class_1, replace=True, n_samples=len(class_0), random_state=42)
        balanced_train_df = pd.concat([class_0, class_1_upsampled])
    else:
        class_0_upsampled = resample(class_0, replace=True, n_samples=len(class_1), random_state=42)
        balanced_train_df = pd.concat([class_0_upsampled, class_1])

    X_train_balanced = balanced_train_df.drop(columns=[label_col])
    y_train_balanced = balanced_train_df[label_col]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train_balanced, y_test, scaler


def run_grid_search(grid, X_train, y_train, X_test, y_test):


    grid.fit(X_train, y_train)
    print(f"Najlepsze parametry dla {grid.estimator.steps[-1][1].__class__.__name__}: {grid.best_params_}")
    print(f"Najlepszy wynik walidacji krzyżowej (CV): {grid.best_score_:.4f}")

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return grid, accuracy, f1


def main():

    print("1. Wczytywanie i przygotowywanie metadanych...")
    dfs = []
    for key, value in all_data.items():
        METADATA_PATH = value['metadata']
        for flac_folder in value['flac']:
            try:
                cols = COLS_DF if key == "DF" else COLS_LA_PA
                df = pd.read_csv(METADATA_PATH, sep=r"\s+", header=None, names=cols, on_bad_lines='warn')
                df = prepare_filepaths(df, flac_folder)

                print(f"Znaleziono {df.shape[0]} pasujących plików dla {key} w {os.path.basename(flac_folder)}")
                if df.empty: continue


                sample_size = 5000 if key == "LA" else 2500
                df = df.sample(min(len(df), sample_size)).copy()
                dfs.append(df)
            except FileNotFoundError:
                print(f"OSTRZEŻENIE: Nie znaleziono pliku metadanych: {METADATA_PATH}")

    if not dfs:
        print("BŁĄD: Nie wczytano żadnych danych. Sprawdź ścieżki i konfigurację.")
        return

    final_df = pd.concat(dfs, ignore_index=True, join='inner')
    print(f"\nŁącznie do przetworzenia: {len(final_df)} plików.")

    feature_extractors = {
        'MFCC': extract_mfcc, 'LFCC': extract_lfcc, 'CQCC': extract_cqcc,
        'GTCC': extract_gtcc, 'WPT': extract_wpt, 'Formants': analyze_formants_and_silence,
    }

    print("\n2. Ekstrakcja cech (proces zrównoleglony)...")
    for name, func in feature_extractors.items():
        print(f"   - Ekstrahuję: {name}")

        results = Parallel(n_jobs=-1, verbose=1)(delayed(func)(path) for path in final_df['file_path'])
        final_df[name] = results


    initial_rows = len(final_df)
    final_df.dropna(subset=list(feature_extractors.keys()), inplace=True)
    print(f"\nUsunięto {initial_rows - len(final_df)} wierszy z powodu błędów ekstrakcji.")
    if final_df.empty:
        print("BŁĄD: Brak poprawnych danych po ekstrakcji cech. Prerywam działanie.")
        return

    print("\n3. Przygotowywanie danych do modelowania...")
    feature_columns = list(feature_extractors.keys())

    new_df = expand_feature_columns(final_df, feature_columns, ['label'])

    X_train_scaled, X_test_scaled, y_train_balanced, y_test, scaler = preprocess_for_modeling(new_df)
    print(f"Rozmiar zbioru treningowego (zbalansowanego): {X_train_scaled.shape}")
    print(f"Rozmiar zbioru testowego: {X_test_scaled.shape}")

    print("\n4. Trenowanie i ocena modeli...")
    if not os.path.exists(CACHE_DIR): os.makedirs(CACHE_DIR)
    memory = Memory(location=CACHE_DIR, verbose=0)

    def create_pipeline(clf):
        return Pipeline([('pca', PCA()), ('clf', clf)], memory=memory)

    pca_components = [10, 20, 30, 40]


    svm_pipeline = create_pipeline(SVC(probability=True))
    svm_params = {'pca__n_components': pca_components, 'clf__C': [1, 10], 'clf__kernel': ['rbf'],
                  'clf__gamma': ['scale']}
    grid_svm = GridSearchCV(svm_pipeline, svm_params, scoring='accuracy', cv=3, n_jobs=-1, verbose=1)

    xgb_pipeline = create_pipeline(XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    xgb_params = {'pca__n_components': pca_components, 'clf__n_estimators': [50, 100], 'clf__max_depth': [3, 5],
                  'clf__learning_rate': [0.1]}
    grid_xgb = GridSearchCV(xgb_pipeline, xgb_params, scoring='accuracy', cv=3, n_jobs=-1, verbose=1)

    print("\n--- Uruchamianie GridSearchCV dla SVM ---")
    grid_svm, svm_accuracy, svm_f1 = run_grid_search(grid_svm, X_train_scaled, y_train_balanced, X_test_scaled, y_test)

    print("\n--- Uruchamianie GridSearchCV dla XGBoost ---")
    grid_xgb, xgb_accuracy, xgb_f1 = run_grid_search(grid_xgb, X_train_scaled, y_train_balanced, X_test_scaled, y_test)

    print("\n" + "=" * 50)
    print("PODSUMOWANIE WYNIKÓW NA DANYCH TESTOWYCH")
    print("=" * 50)
    print(f"\nModel SVM:")
    print(f"  - Dokładność (Accuracy): {svm_accuracy:.4f}")
    print(f"  - Wynik F1 (F1 Score):   {svm_f1:.4f}")
    print(f"\nModel XGBoost:")
    print(f"  - Dokładność (Accuracy): {xgb_accuracy:.4f}")
    print(f"  - Wynik F1 (F1 Score):   {xgb_f1:.4f}")
    print("=" * 50)
    print("\nSkrypt zakończył działanie.")


if __name__ == "__main__":
    main()