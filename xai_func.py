import pickle
import joblib
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from ASV_dl_func import extract_features, collate_fn_padd, AudioDataset, prepare_data_GMM_BiLSTM, transpose_cqcc, \
    load_gmm_models, load_bilstm_model, eval_model, fused_score
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def train_gridsearch(X, y, param_grid=None):
    if param_grid is None:
        param_grid = {
            'C': [0.1, 1, 10, 100, 1000],
            'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['rbf']
        }

    grid = GridSearchCV(
        estimator=SVC(probability=True),
        param_grid=param_grid,
        refit=True,
        verbose=4,
        n_jobs=-1
    )

    grid.fit(X, y)

    return grid

def prepare_pertubation_data_GMM_BiLSTM(
    df,
    feature_map,
    label_col='label',
    feature_col="cqcc",
    model_dir="GMM-BiLSTM",
    scaler_file="scaler.pkl",
    transpose_func=transpose_cqcc,
    list_model_gmm=None,
    perturb_mode="feature",
    n_samples=10,
    min_feats=0,
    max_feats=18,
    num_slices=21,
    prob_active=0.45,
    train_regression=False
):
    if list_model_gmm is None:
        list_model_gmm = ["gmm_genuine.pkl", "gmm_df.pkl"]

    df = extract_features(df, feature_map)
    df = prepare_data_GMM_BiLSTM(
        df,
        label_col=label_col,
        feature_col=feature_col,
        transpose_func=transpose_func
    )

    scaler = joblib.load(os.path.join(model_dir, scaler_file))
    df[feature_col] = df[feature_col].apply(lambda x: scaler.transform(x))

    perturbed_info = []
    perturbation_matrix = []
    predictions_matrix = []
    df = df.reset_index(drop=True)
    for i in range(len(df)):
        data = df.loc[i, feature_col]
        original_shape = data.shape

        if original_shape[0] == 19 and original_shape[1] == 63:
            feat_axis, time_axis = 0, 1
        elif original_shape[0] == 63 and original_shape[1] == 19:
            feat_axis, time_axis = 1, 0
        else:
            raise ValueError(f"Nieoczekiwany kształt cech CQCC: {original_shape}")

        if perturb_mode == "feature":
            perturbed, changed = perturb_random_features_mean(
                data.T, min_feats=min_feats, max_feats=max_feats
            )
            perturbed = perturbed.T

            diff = np.any(data != perturbed, axis=time_axis)
            assert np.all(diff[changed] == True), "Nie wszystkie wybrane cechy zostały zmienione!"
            assert np.all(diff[np.setdiff1d(np.arange(19), changed)] == False), "Zmieniono niewłaściwe cechy!"

            df.at[i, feature_col] = np.array(perturbed, dtype=object)
            perturbed_info.append({
                "mode": "feature",
                "changed_feats": changed.tolist(),
                "shape_check": perturbed.shape
            })

            if train_regression:
                perturbation_matrix.append(np.any(data != perturbed, axis=time_axis).astype(int))

        elif perturb_mode == "time":
            perturbed, active_vec = perturb_segments(
                data.T,
                num_slices=num_slices,
                axis=1,
                prob_active=prob_active
            )
            perturbed = perturbed.T

            df.at[i, feature_col] = np.array(perturbed, dtype=object)
            perturbed_info.append({
                "mode": "time",
                "active_segments": active_vec.astype(int).tolist(),
                "shape_check": perturbed.shape
            })

            if train_regression:
                perturbation_matrix.append(active_vec.astype(int))

        else:
            raise ValueError(f"Nieznany tryb perturbacji: {perturb_mode}")

    df[feature_col] = df[feature_col].apply(lambda x: np.asarray(x, dtype=np.float32))

    audio_dataset = AudioDataset(df)
    audio_loader = DataLoader(
        audio_dataset,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn_padd
    )

    input_dim = df[feature_col].iloc[0].shape[1]
    bilstm_model = load_bilstm_model(input_dim=input_dim)

    y_true, y_pred, metrics = eval_model(
        bilstm_model,
        train_df=None,
        test_df=df,
        test_loader=audio_loader,
        feature_name=feature_col,
        label_name=label_col,
        use_saved_models=True,
        verbose=False,
        list_model_gmm=list_model_gmm
    )

    if train_regression:
        perturbation_matrix = np.array(perturbation_matrix)
        predictions_matrix = np.array([y_pred[i] for i in range(len(perturbation_matrix))])

        cosine_distances = pairwise_distances(
            perturbation_matrix,
            np.ones((1, perturbation_matrix.shape[1])),
            metric='cosine'
        ).ravel()
        weights = np.sqrt(np.exp(-(cosine_distances ** 2) / 0.25 ** 2))

        explainable_model = LinearRegression()
        target_values = predictions_matrix
        explainable_model.fit(perturbation_matrix, target_values, sample_weight=weights)

        os.makedirs("xai_model", exist_ok=True)
        model_name = f"xai_model/lime_regression_{perturb_mode}_slice_{num_slices}.pkl"
        joblib.dump(explainable_model, model_name)
        print(f"Zapisano model regresji LIME w: {model_name}")

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "perturbed_info": perturbed_info
    }

def test_lime_on_single_instance(
        instance_cqcc,  # (19, 63)
        perturb_mode="feature",
        num_slices=9,
        n_perturbations=20,
        model_path=None,
        model_dir="GMM-BiLSTM",
        list_model_gmm=None,
        label_col="label",
        feature_col="cqcc",
        transpose_func=None,
        scaler_file="scaler.pkl",
        train_instance=True
):
    if list_model_gmm is None:
        list_model_gmm = ["gmm_genuine.pkl", "gmm_df.pkl"]

    if transpose_func is None:
        transpose_func = lambda x: x.T

    df = pd.DataFrame({feature_col: [instance_cqcc], label_col: [0]})
    df = prepare_data_GMM_BiLSTM(
        df,
        label_col=label_col,
        feature_col=feature_col,
        transpose_func=transpose_func
    )

    scaler = joblib.load(os.path.join(model_dir, scaler_file))
    df[feature_col] = df[feature_col].apply(lambda x: scaler.transform(x))

    instance_cqcc = df[feature_col].iloc[0]
    input_dim = instance_cqcc.shape[1]

    bilstm_model = load_bilstm_model(input_dim=input_dim)
    _, gmm_genuine, gmm_df = load_gmm_models(model_dir)

    if train_instance:
        n_perturbations_to_use = 30
        perturbation_matrix = []
        y_pred = []

        for _ in range(n_perturbations_to_use):
            if perturb_mode == "feature":

                perturbed, _ = perturb_random_features_mean(
                    instance_cqcc.T,
                    min_feats=0,
                    max_feats=instance_cqcc.shape[0] - 1
                )
                perturbed = perturbed.T
                perturb_vector = np.any(instance_cqcc != perturbed, axis=1).astype(int)

            elif perturb_mode == "time":
                perturbed, active_vec = perturb_segments(
                    instance_cqcc.T,
                    num_slices=num_slices,
                    axis=1,
                    prob_active=0.5
                )
                perturbed = perturbed.T
                perturb_vector = active_vec.astype(int)
            else:
                raise ValueError(f"Nieznany tryb perturbacji: {perturb_mode}")

            perturbation_matrix.append(perturb_vector)

            sample_x_tensor = torch.tensor(perturbed, dtype=torch.float32)
            sample_features_np = perturbed

            score = fused_score(
                bilstm_model,
                sample_x_tensor,
                sample_features_np,
                gmm_genuine,
                gmm_df
            )
            y_pred.append(1 if score > 0.5 else 0)

        perturbation_matrix = np.array(perturbation_matrix)
        predictions_matrix = np.array(y_pred)

        cosine_distances = pairwise_distances(
            perturbation_matrix,
            np.ones((1, perturbation_matrix.shape[1])),
            metric='cosine'
        ).ravel()
        weights = np.sqrt(np.exp(-(cosine_distances ** 2) / 0.25 ** 2))

        explainable_model = LinearRegression()
        explainable_model.fit(perturbation_matrix, predictions_matrix, sample_weight=weights)

    else:
        if model_path is None:
            model_path = f"xai_model/lime_regression_{perturb_mode}_slice_{num_slices}.pkl"
        explainable_model = joblib.load(model_path)

    sample_x_tensor = torch.tensor(instance_cqcc, dtype=torch.float32)
    sample_features_np = instance_cqcc

    score = fused_score(
        bilstm_model,
        sample_x_tensor,
        sample_features_np,
        gmm_genuine,
        gmm_df
    )
    y_pred = 1 if score > 0.5 else 0

    lime_coeffs = explainable_model.coef_
    top_indices = np.argsort(np.abs(lime_coeffs))[-5:][::-1]

    return {
        "lime_coeffs": lime_coeffs,
        "top_indices": top_indices,
        "predicted_label": y_pred,
        "model_score": score,
        "explainable_model": explainable_model
    }
def visualize_cqcc_perturbation(signal_original, perturbed_info, index=0, num_slices=21):

    info = perturbed_info[index]
    mode = info["mode"]

    signal_perturbed = np.copy(signal_original)
    num_features, num_frames = signal_original.shape

    if mode == "feature":
        changed_feats = info.get("changed_feats", [])
        for feat in changed_feats:

            mean_val = np.mean(signal_original[feat, :])
            signal_perturbed[feat, :] = mean_val

    elif mode == "time":
        active_segments = np.array(info.get("active_segments", []))

        slice_length = int(np.ceil(num_frames / num_slices))
        for slice_idx, active in enumerate(active_segments):
            if active:
                start = slice_idx * slice_length
                end = min((slice_idx + 1) * slice_length, num_frames)

                segment_mean = np.mean(signal_original[:, start:end], axis=0)
                signal_perturbed[:, start:end] = segment_mean

    else:
        raise ValueError(f"Nieznany tryb perturbacji: {mode}")

    plt.figure(figsize=(15, 2.5 * num_features))
    plt.suptitle(f"CQCC perturbacje (tryb: {mode})", fontsize=14, fontweight='bold')

    for feat_idx in range(num_features):
        plt.subplot(num_features, 1, feat_idx + 1)
        plt.plot(signal_original[feat_idx, :], label="oryginał", color='black', linewidth=1.2)
        plt.plot(signal_perturbed[feat_idx, :], label="po perturbacji", color='red', alpha=0.7, linewidth=1)

        if mode == "feature":
            changed_feats = info.get("changed_feats", [])
            if feat_idx in changed_feats:
                plt.gca().set_facecolor((1.0, 0.9, 0.9))

        elif mode == "time":
            active_segments = np.array(info.get("active_segments", []))
            slice_length = int(np.ceil(num_frames / num_slices))
            for slice_idx, active in enumerate(active_segments):
                if active:
                    start = slice_idx * slice_length
                    end = min((slice_idx + 1) * slice_length, num_frames)
                    plt.axvspan(start - 0.5, end - 0.5, color='red', alpha=0.15)

        plt.title(f"CQCC cecha {feat_idx}")
        plt.xlabel("Ramka czasowa")
        plt.ylabel("Amplituda")
        plt.grid(True, linestyle='--', linewidth=0.5)
        if feat_idx == 0:
            plt.legend(loc="upper right", fontsize=8)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


def calculate_cosine_distances_time(random_perturbations, num_slices):

    original_ecg_rep = np.ones((1, num_slices))
    cosine_distances = pairwise_distances(random_perturbations, original_ecg_rep, metric='cosine').ravel()

    return cosine_distances

def calculate_cosine_distances_feature_mean(original_data, min_feats=1, max_feats=9):

    num_features, num_slices = original_data.shape
    perturbed_data = original_data.copy()

    num_to_perturb = np.random.randint(min_feats, min(max_feats, num_features) + 1)
    chosen_feats = np.random.choice(num_features, num_to_perturb, replace=False)

    for idx in chosen_feats:
        perturbed_data[idx, :] = np.mean(perturbed_data[idx, :])

    original_flat = original_data.flatten()[np.newaxis, :]
    perturbed_flat = perturbed_data.flatten()[np.newaxis, :]
    cosine_distance = pairwise_distances(perturbed_flat, original_flat, metric='cosine').ravel()[0]

    return cosine_distance, chosen_feats, perturbed_data


def calculate_weights_from_distances(cosine_distances, kernel_width=0.25):

    weights = np.sqrt(np.exp(-(cosine_distances ** 2) / kernel_width ** 2))
    return weights

def scale_data(df_train, df_test, col_name):

    df_scaled = {"train": df_train.copy(), "test": df_test.copy()}

    scaler = StandardScaler()
    df_scaled['train'][col_name] = scaler.fit_transform(df_scaled['train'][col_name])
    df_scaled['train'][col_name] = scaler.fit(df_scaled['train'][col_name])

    return scaler, df_scaled


def expand_selected_features(df, features):
    df = df.copy()
    for feature in features:
        if feature not in df.columns:
            print(f"Kolumna '{feature}' nie istnieje — pomijam.")
            continue
        df = df[df[feature].notna()].reset_index(drop=True)
        first_val = df[feature].iloc[0]
        if not hasattr(first_val, "__len__"):
            print(f"Kolumna '{feature}' nie zawiera listy/ndarray — pomijam.")
            continue
        feature_len = len(first_val)
        expanded = pd.DataFrame(
            df[feature].to_list(),
            columns=[f"{feature}_{i+1}" for i in range(feature_len)]
        )
        df = pd.concat([df.drop(columns=[feature]), expanded], axis=1)
        print(f"Rozdzielono kolumnę '{feature}' na {feature_len} podkolumn.")
    return df

def signal_segmentation(data, num_slices= 21, axis = 1):
    length = data.shape[axis]
    remainder = length % num_slices
    usable = length - remainder

    if remainder > 0:
        slicer = [slice(None)] * data.ndim
        slicer[axis] = slice(0, usable)
        data = data[tuple(slicer)]

    parts = np.split(data, num_slices, axis=axis)
    return parts, remainder

def perturb_segments(data, num_slices=21, axis=1, prob_active=0.45):
    parts, remainder = signal_segmentation(data, num_slices, axis)

    active_vec = np.random.rand(num_slices) < prob_active
    perturbed_parts = []

    for active, p in zip(active_vec, parts):
        if active:
            mean_val = np.mean(p, axis=axis, keepdims=True)
            new_part = np.repeat(mean_val, p.shape[axis], axis=axis)
            perturbed_parts.append(new_part)
        else:
            perturbed_parts.append(p)

    perturbed_data = np.concatenate(perturbed_parts, axis=axis)
    return perturbed_data, active_vec

def perturb_random_features_mean(data, min_feats=0, max_feats=18):

    num_features = data.shape[0]
    perturbed = data.copy()

    num_to_perturb = np.random.randint(min_feats, min(max_feats, num_features) + 1)
    chosen_indices = np.random.choice(num_features, num_to_perturb, replace=False)

    for idx in chosen_indices:
        mean_val = np.mean(perturbed[idx, :])
        perturbed[idx, :] = mean_val

    return perturbed, chosen_indices

def plot_cqcc_pipeline(df, perturbed_info, feature_col="cqcc", idx=0):
    original_data = df.loc[idx, feature_col].copy()
    perturbed_data = df.loc[idx, feature_col]
    info = perturbed_info[idx]

    if info["mode"] == "time":
        scales = np.array(info["active_segments"])
        plot_cqcc_perturbations(
            original_data.T,
            perturbed_data.T,
            scales=scales,
            title=f"CQCC – perturbacje czasowe próbki {idx}",
            time=True
        )
    elif info["mode"] == "feature":
        scales = np.array(info["changed_feats"])
        plot_cqcc_perturbations(
            original_data.T,
            perturbed_data.T,
            scales=scales,
            title=f"CQCC – perturbacje cech próbki {idx}",
            time=False
        )


def plot_cqcc_perturbations(original, perturbed, scales, title="CQCC z perturbacjami", time=True):
    num_coeffs, total_len = original.shape
    time_axis = np.arange(total_len)

    fig, axes = plt.subplots(num_coeffs, 1, figsize=(12, 2 * num_coeffs), sharex=True)
    if num_coeffs == 1:
        axes = [axes]

    for i in range(num_coeffs):
        axes[i].plot(time_axis, original[i], label="oryginał", alpha=0.6)
        axes[i].plot(time_axis, perturbed[i], label="po perturbacji", alpha=0.8)

        if time:
            if scales is not None:
                num_slices = len(scales)
                seg_len = total_len // num_slices
                for j, s in enumerate(scales):
                    if s != 1.0 and s != 0.0:
                        start = j * seg_len
                        end = (j + 1) * seg_len
                        color = "red" if s > 1 else "blue"
                        axes[i].axvspan(start, end, color=color, alpha=0.15)

        else:
            if scales is not None and i in scales:
                axes[i].axhline(np.mean(original[i]), color='red', linestyle='--', alpha=0.5)
                axes[i].set_facecolor((1, 0.9, 0.9))

    if not time and scales is not None:
        print(f"Uśrednione cechy: {scales}")

    axes[-1].set_xlabel("czas (próbki)")
    axes[0].legend(loc="upper right")
    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

