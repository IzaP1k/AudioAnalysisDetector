import pandas as pd
import shap
from ASV_dl_func import load_gmm_models, load_bilstm_model, compute_llr
import numpy as np
import matplotlib.pyplot as plt
import torch
from captum.attr import IntegratedGradients, Occlusion
import joblib


def preprocess_signal(instance_signal, feature_col, label_col,
                      scaler_path, model_dir, transpose_func, prepare_func):
    df = pd.DataFrame({feature_col: [instance_signal], label_col: [0]})
    df = prepare_func(df, label_col=label_col, feature_col=feature_col, transpose_func=transpose_func)

    scaler = joblib.load(scaler_path)
    df[feature_col] = df[feature_col].apply(lambda x: scaler.transform(x))

    instance_preprocessed = df[feature_col].iloc[0]
    T, F = instance_preprocessed.shape

    _, gmm_genuine, gmm_df = load_gmm_models(model_dir)
    bilstm_model = load_bilstm_model(input_dim=F)

    return instance_preprocessed, T, F, gmm_genuine, gmm_df, bilstm_model


def compute_gmm_shap(instance_data, gmm_genuine, gmm_df,
                     window_size=10, stride=5, background_windows=10):
    T, F = instance_data.shape
    windows = []
    start_indices = []

    for start in range(0, T - window_size + 1, stride):
        window_flat = instance_data[start:start + window_size, :].flatten()
        windows.append(window_flat)
        start_indices.append(start)

    windows = np.array(windows)
    X_background = windows[:background_windows]

    def shap_wrapper(windows_batch):
        results = []
        for i, win in enumerate(windows_batch):
            start = start_indices[i % len(start_indices)]
            data_copy = instance_data.copy()
            data_copy[start:start + window_size, :] = win.reshape(window_size, F)
            gmm_llr = compute_llr(data_copy, gmm_genuine, gmm_df)
            gmm_prob = 1 / (1 + np.exp(-gmm_llr))
            results.append(gmm_prob)
        return np.array(results)

    explainer = shap.KernelExplainer(shap_wrapper, X_background)
    shap_values = explainer.shap_values(windows)

    heatmap = np.zeros_like(instance_data, dtype=float)
    count_map = np.zeros_like(instance_data, dtype=float)
    for i, start in enumerate(start_indices):
        shap_window = shap_values[i].reshape(window_size, F)
        heatmap[start:start + window_size, :] += shap_window
        count_map[start:start + window_size, :] += 1

    np.divide(heatmap, count_map, out=heatmap, where=(count_map != 0))

    heatmap = np.nan_to_num(heatmap, nan=0.0, posinf=0.0, neginf=0.0)

    return heatmap


def compute_bilstm_heatmaps(instance_data, bilstm_model, device=None, target=1):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bilstm_model.to(device).eval()
    input_tensor = torch.tensor(instance_data[np.newaxis, :, :], dtype=torch.float32).to(device)

    ig = IntegratedGradients(bilstm_model)
    baseline = torch.zeros_like(input_tensor)
    attr_ig = ig.attribute(input_tensor, baselines=baseline, target=target)
    attr_ig_np = attr_ig.squeeze().cpu().numpy()

    occlusion = Occlusion(bilstm_model)
    attr_occ = occlusion.attribute(input_tensor,
                                   sliding_window_shapes=(5, 1),
                                   target=target)
    attr_occ_np = attr_occ.squeeze().cpu().numpy()

    feature_importance = np.mean(np.abs(attr_ig_np), axis=0)
    time_importance = np.sum(np.abs(attr_occ_np), axis=1)

    return attr_ig_np, attr_occ_np, feature_importance, time_importance


def compare_gmm_bilstm(instance_signal, feature_col, label_col,
                       scaler_path, model_dir, transpose_func, prepare_func):
    instance_data, T, F, gmm_genuine, gmm_df, bilstm_model = preprocess_signal(
        instance_signal, feature_col, label_col, scaler_path, model_dir,
        transpose_func, prepare_func
    )

    heatmap_gmm = compute_gmm_shap(instance_data, gmm_genuine, gmm_df)

    attr_ig_np, attr_occ_np, feature_importance, time_importance = compute_bilstm_heatmaps(instance_data, bilstm_model)

    return {
        "gmm_shap": heatmap_gmm,
        "bilstm_ig": attr_ig_np,
        "bilstm_occlusion": attr_occ_np,
    }, {"feature_importance": feature_importance,
        "time_importance": time_importance}


def plot_heatmaps_separately_stylish(heatmaps_dict):
    for name, heatmap in heatmaps_dict.items():
        plt.figure(figsize=(12, 5))
        plt.imshow(heatmap.T, aspect='auto', origin='lower', cmap='coolwarm', alpha=0.9)

        plt.title(name, fontsize=16, fontweight='bold')
        plt.xlabel('Czas [frame]', fontsize=12)
        plt.ylabel('Cecha', fontsize=12)

        cbar = plt.colorbar(label='Wpływ cechy')
        cbar.ax.tick_params(labelsize=10)
        cbar.outline.set_linewidth(1)

        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show()


def extract_top_intervals_global(signal_original, heatmap,
                                 threshold=1e-2,
                                 sample_rate=16000,
                                 hop_length=512,
                                 min_duration=0.2,
                                 top_n=5):
    if heatmap.shape != signal_original.shape:
        heatmap = heatmap.T

    num_features, num_frames = signal_original.shape
    time_per_frame = hop_length / sample_rate

    all_pos = []
    all_neg = []

    for feat_idx in range(num_features):
        nonzero_times = np.where(np.abs(heatmap[feat_idx, :]) > threshold)[0]
        if len(nonzero_times) == 0:
            continue

        groups = np.split(nonzero_times, np.where(np.diff(nonzero_times) > 1)[0] + 1)
        for g in groups:
            values = heatmap[feat_idx, g]
            values = values[~np.isnan(values)]
            values = values[values != 0]

            if len(values) == 0:
                continue

            start_t = g[0] * time_per_frame
            end_t = (g[-1] + 1) * time_per_frame
            duration = end_t - start_t
            if duration < min_duration:
                continue

            mean_val = float(np.mean(values))
            interval_info = {
                "feature": f"F{feat_idx}",
                "start": round(start_t, 3),
                "end": round(end_t, 3),
                "strength": mean_val
            }

            if mean_val > 0:
                all_pos.append(interval_info)
            elif mean_val < 0:
                all_neg.append(interval_info)

    top_strongest = sorted(all_pos, key=lambda x: x["strength"], reverse=True)[:top_n]
    top_weakest = sorted(all_neg, key=lambda x: x["strength"])[:top_n]


    return {
        "strongest": top_strongest,
        "weakest": top_weakest
    }


"""
result = compare_gmm_bilstm(
    instance_signal=instance_cqcc_original,
    feature_col='cqcc',
    label_col='label',
    scaler_path=os.path.join(model_dir, scaler_file),
    model_dir=model_dir,
    transpose_func=transpose_func,
    prepare_func=prepare_data_GMM_BiLSTM
)

plot_heatmaps(result)

"""
