import glob
import os
import torch
import joblib
import numpy as np
import matplotlib.pyplot as plt
from captum.attr import Occlusion
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


def load_resources(model_folder, scaler_filename):
    path = os.path.join(model_folder, scaler_filename)
    try:
        return joblib.load(path)
    except FileNotFoundError:
        print(f"Błąd: Nie znaleziono pliku scalera w ścieżce: {path}")
        return None


def load_model_weights(model_class, weight_path, device='cpu'):
    model = model_class(num_classes=2)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model


def preprocess_spectrogram(spectrogram_raw, scaler):
    x = spectrogram_raw.numpy() if isinstance(spectrogram_raw, torch.Tensor) else spectrogram_raw
    x_scaled = scaler.transform(x)

    if x_scaled.ndim == 1:
        x_scaled = x_scaled[np.newaxis, :, np.newaxis]
    elif x_scaled.ndim == 2:
        x_scaled = x_scaled[np.newaxis, :, :]

    input_tensor = torch.tensor(x_scaled, dtype=torch.float32).unsqueeze(0)
    input_tensor.requires_grad = True

    return input_tensor


def compute_xai_attributes(model, input_tensor, target_class):

    target_layers = [model.residual_blocks[5]]
    targets = [ClassifierOutputTarget(target_class)]

    with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
        attr_gc_upsampled = grayscale_cam[0, :]

    occlusion = Occlusion(model)

    attr_occ = occlusion.attribute(input_tensor,
                                   strides=(1, 4, 4),
                                   target=target_class,
                                   sliding_window_shapes=(1, 15, 15),
                                   baselines=0)

    attr_occ_upsampled = torch.nn.functional.interpolate(
        attr_occ, size=input_tensor.shape[2:], mode='bilinear', align_corners=False
    )

    return attr_gc_upsampled, attr_occ_upsampled.detach().cpu().numpy()[0, 0]


def analyze_time_focus(mask_energy, W, total_energy):
    one_third = W // 3
    time_regions = {
        "Początek nagrania": mask_energy[:, :one_third],
        "Środek (główna treść)": mask_energy[:, one_third:2 * one_third],
        "Końcówka nagrania": mask_energy[:, 2 * one_third:]
    }
    time_sums = {k: np.sum(v) for k, v in time_regions.items()}
    best_time_key = max(time_sums, key=time_sums.get)
    best_time_pct = (time_sums[best_time_key] / total_energy) * 100
    return best_time_key, best_time_pct


def analyze_freq_focus(mask_energy, H, total_energy):
    band_h = H // 3
    freq_regions = {
        "Niskie tony (Bas/Tło)": (mask_energy[:band_h, :],
                                  "Model nasłuchiwał tła, buczenia mikrofonu lub głębi głosu."),
        "Średnie tony (Mowa)": (mask_energy[band_h:2 * band_h, :],
                                "Model skupił się na brzmieniu głosu i wymawianych słowach."),
        "Wysokie tony (Szum/Detale)": (mask_energy[2 * band_h:, :],
                                       "Model szukał cyfrowych artefaktów, szumów lub 'syczenia' typowego dla manipulacji głosowych.")
    }
    freq_sums = {k: np.sum(v[0]) for k, v in freq_regions.items()}
    best_freq_key = max(freq_sums, key=freq_sums.get)
    best_freq_pct = (freq_sums[best_freq_key] / total_energy) * 100
    freq_explanation = freq_regions[best_freq_key][1]
    return best_freq_key, best_freq_pct, freq_explanation


def determine_focus_strength(best_freq_pct):
    if best_freq_pct > 50:
        return "bardzo silnie"
    elif best_freq_pct > 30:
        return "umiarkowanie"
    return "równomiernie"


def format_explanation_report(best_time_key, best_time_pct, focus_strength, best_freq_key, freq_explanation,
                              best_freq_pct):
    return (
        f"Analiza słowna predykcji modelu\n"
        f"\n\n"
        f"Kiedy model analizuje?\n"
        f"   Decyzja zapadła głównie przez analizę sekcji: '{best_time_key}'.\n"
        f"   Stanowi to {best_time_pct:.1f}% całej uwagi modelu.\n\n"
        f"Na jakiej wysokości dźwięku skupia się model?\n"
        f"   Model {focus_strength} skupił się na pasmie: {best_freq_key}.\n"
        f"   Co daje {freq_explanation} znaczenie dla modelu\n"
        f"   To pasmo ma następującą siłę wpływu: {best_freq_pct:.1f}%."
    )


def generate_human_friendly_explanation(mask):
    mask_energy = np.abs(mask)
    H, W = mask_energy.shape
    total_energy = np.sum(mask_energy) + 1e-9

    best_time_key, best_time_pct = analyze_time_focus(mask_energy, W, total_energy)
    best_freq_key, best_freq_pct, freq_explanation = analyze_freq_focus(mask_energy, H, total_energy)
    focus_strength = determine_focus_strength(best_freq_pct)

    return format_explanation_report(
        best_time_key, best_time_pct, focus_strength,
        best_freq_key, freq_explanation, best_freq_pct
    )


def plot_input_spectrogram(ax, original_img):
    ax.imshow(original_img, origin='lower', aspect='auto', cmap='gray')
    ax.set_title("1. Wejściowy Spektrogram", fontsize=12)
    ax.set_ylabel("Częstotliwość")
    ax.set_xlabel("Czas")


def plot_gradcam_overlay(ax, fig, original_img, gradcam_mask):
    ax.imshow(original_img, origin='lower', aspect='auto', cmap='gray', alpha=1.0)
    gradcam_mask = np.maximum(gradcam_mask, 0)
    im = ax.imshow(gradcam_mask, origin='lower', aspect='auto', cmap='jet', alpha=0.5)
    ax.set_title("Metoda Grad-CAM++", fontsize=12)
    ax.set_yticks([])
    ax.set_xlabel("Czas")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Siła aktywacji (Wagi)', rotation=270, labelpad=15)


def plot_occlusion_overlay(ax, fig, original_img, occ_mask):
    ax.imshow(original_img, origin='lower', aspect='auto', cmap='gray', alpha=1.0)
    limit = np.percentile(np.abs(occ_mask), 99.5)
    im = ax.imshow(occ_mask, origin='lower', aspect='auto', cmap='seismic',
                   vmin=-limit, vmax=limit, alpha=0.7)
    ax.set_title("Metoda Occlusion (Test przez zasłanianie)", fontsize=12)
    ax.set_yticks([])
    ax.set_xlabel("Czas")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Wpływ na decyzję (Czerwony=Ważne)', rotation=270, labelpad=15)


def plot_xai_dashboard_styled(original_img, gradcam_mask, occ_mask, title_info):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    plt.suptitle(title_info, fontsize=16, fontweight='bold')

    plot_input_spectrogram(axes[0], original_img)
    plot_gradcam_overlay(axes[1], fig, original_img, gradcam_mask)
    plot_occlusion_overlay(axes[2], fig, original_img, occ_mask)

    plt.tight_layout()
    plt.show()


def resolve_paths(model_folder, model_files, scaler_file):
    if model_files is None:
        model_paths = glob.glob(os.path.join(model_folder, "*.pt"))
        model_files = [os.path.basename(p) for p in model_paths]

    if scaler_file is None:
        scaler_paths = glob.glob(os.path.join(model_folder, "*.pkl"))
        scaler_file = os.path.basename(scaler_paths[0]) if scaler_paths else None

    return model_files, scaler_file


def prepare_input_data(data_row, col_name, label, model_folder, scaler_file):
    scaler = load_resources(model_folder, scaler_file)
    raw_spect = data_row[col_name]
    true_label = data_row[label]
    input_tensor = preprocess_spectrogram(raw_spect, scaler)
    return input_tensor, raw_spect, true_label


def run_model_inference(model_class, model_folder, model_name, input_tensor):
    full_path = os.path.join(model_folder, model_name)
    model = load_model_weights(model_class, full_path)

    if model is None:
        return None, None, None

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_idx].item()

    return model, pred_idx, confidence


def display_model_results(model_name, pred_idx, confidence, true_label, model, input_tensor, original_img_npy):
    print(f"Wynik modelu: Klasa {pred_idx} (Pewność: {confidence:.1%})")

    gradcam, occ = compute_xai_attributes(model, input_tensor, target_class=pred_idx)

    print("\nGrad-CAM++ (Ogólna)")
    print(generate_human_friendly_explanation(gradcam))

    print("\nOcclusion Sensitivity (Test przez zasłanianie)")
    print(generate_human_friendly_explanation(np.maximum(occ, 0)))

    plot_title = f"Model: {model_name} | Predykcja: {pred_idx} ({confidence:.0%}) | Prawda: {true_label}"
    plot_xai_dashboard_styled(original_img_npy, gradcam, occ, plot_title)


def process_single_model(model_name, model_folder, model_class, input_tensor, original_img_npy, true_label):
    print(f"\n{'=' * 60}\n MODEL: {model_name}\n{'=' * 60}")

    model, pred_idx, confidence = run_model_inference(model_class, model_folder, model_name, input_tensor)

    if model is not None:
        display_model_results(model_name, pred_idx, confidence, true_label, model, input_tensor, original_img_npy)


def run_analysis_pipeline(data_row, model_folder, model_files, model_class, scaler_file, col_name='mel-spect',
                          label='label'):
    model_files, scaler_file = resolve_paths(model_folder, model_files, scaler_file)
    input_tensor, raw_spect, true_label = prepare_input_data(data_row, col_name, label, model_folder, scaler_file)

    if input_tensor is None:
        return

    original_img_npy = input_tensor.detach().cpu().numpy()[0, 0]
    print(f"\n>>> Analiza próbki (Prawdziwa Etykieta: {true_label}) | Rozmiar: {raw_spect.shape}")

    for model_name in model_files:
        process_single_model(model_name, model_folder, model_class, input_tensor, original_img_npy, true_label)