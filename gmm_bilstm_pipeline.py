import os
import pickle
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_curve
from sklearn.mixture import GaussianMixture
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from pytorch_func import AudioDataset, BiLSTMClassifier, collate_fn_padd


def load_gmm_models(model_dir, ubm_model= "ubm.pkl", gmm_genuine_model="gmm_genuine.pkl", gmm_df_name="gmm_df.pkl"):
    print("Wczytywanie zapisanych modeli GMM...")
    with open(os.path.join(model_dir, ubm_model), "rb") as f:
        ubm = pickle.load(f)
    with open(os.path.join(model_dir, gmm_genuine_model), "rb") as f:
        gmm_genuine = pickle.load(f)
    with open(os.path.join(model_dir, gmm_df_name), "rb") as f:
        gmm_df = pickle.load(f)
    print("Modele GMM pomyślnie wczytane.")
    return ubm, gmm_genuine, gmm_df

def gmm_model(train_df, N_COMPONENTS_GMM=128, feature_name='cqcc', label_name="label_num", model_dir="GMM-BiLSTM"):
    os.makedirs(model_dir, exist_ok=True)
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

    with open(os.path.join(model_dir, "ubm.pkl"), "wb") as f:
        pickle.dump(ubm, f)
    with open(os.path.join(model_dir, "gmm_genuine.pkl"), "wb") as f:
        pickle.dump(gmm_genuine, f)
    with open(os.path.join(model_dir, "gmm_df.pkl"), "wb") as f:
        pickle.dump(gmm_df, f)
    print(f"Modele GMM zapisane w folderze '{model_dir}/'.")

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


def eval_model(model, train_df, test_df, test_loader, feature_name: str = 'cqcc',
               label_name: str = "label", model_dir = "GMM-BiLSTM", use_saved_models=True, verbose=True,
               list_model_gmm=None):

    if list_model_gmm is None:
        list_model_gmm = ["gmm_genuine.pkl", "gmm_df.pkl"]
    if use_saved_models and all(os.path.exists(os.path.join(model_dir, f)) for f in list_model_gmm):
        _, gmm_genuine, gmm_df = load_gmm_models(model_dir)
    else:
        if train_df is None:
            raise ValueError("train_df potrzebne do trenowania GMM, jeśli use_saved_models=False")
        gmm_genuine, gmm_df = gmm_model(train_df, feature_name=feature_name, label_name=label_name)

    y_true, y_pred, scores = [], [], []
    start_time_eval = time.time()

    for batch_idx, (X_batch, y_batch) in enumerate(test_loader):
        for i in range(X_batch.size(0)):
            sample_x_tensor = X_batch[i]
            mask = (sample_x_tensor.sum(dim=1) != 0)
            sample_features_np = sample_x_tensor[mask].cpu().numpy()
            score = fused_score(model, sample_x_tensor, sample_features_np, gmm_genuine, gmm_df) \
                if sample_features_np.size else 0.5
            scores.append(score)
            y_pred.append(1 if score > 0.5 else 0)
        y_true.extend(y_batch.numpy())

    end_time_eval = time.time()
    if verbose:
        print(f"Ewaluacja zakończona w {end_time_eval - start_time_eval:.2f} sekund.")

    from sklearn.metrics import accuracy_score, f1_score, roc_curve
    import numpy as np

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute(fnr - fpr))]
    eer = fpr[np.nanargmin(np.absolute(fnr - fpr))]

    if verbose:
        print("\n--- Wyniki końcowe ---")
        print("Accuracy:", accuracy)
        print("F1:", f1)
        print("EER:", eer)

    metrics = {"accuracy": accuracy, "f1": f1, "eer": eer}
    return y_true, y_pred, metrics



def load_bilstm_model(input_dim, model_path):
    model = BiLSTMClassifier(input_dim=input_dim)
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Wczytano model z: {model_path}")
    return model

def get_input_dim_from_loader(data_loader):
    for batch in data_loader:
        inputs = batch[0]  # zakładam, że batch = (inputs, labels)
        input_dim = inputs.shape[2]  # (batch_size, seq_len, feature_dim)
        return input_dim
    raise ValueError("DataLoader jest pusty, nie można określić input_dim")

def evaluate_all_models(bilstm_model="biLstm_best_model.pt", base_dir="GMM-BiLSTM", train_loader=None, train_df=None, test_df=None, test_loader=None):
    if train_loader is None:
        raise ValueError("Podaj train_loader, aby wyznaczyć input_dim.")

    input_dim = get_input_dim_from_loader(train_loader)

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if not os.path.isdir(folder_path):
            continue

        model_path = os.path.join(folder_path, bilstm_model)
        if not os.path.exists(model_path):
            print(f"Brak pliku modelu w {folder_path}")
            continue

        model = load_bilstm_model(input_dim, model_path)
        print(f"\n--- Ewaluacja modelu: {folder} ---")
        eval_model(model, train_df, test_df, test_loader, use_saved_models=False)


def setup_directories(model_dir, optimizer_name, criterion_name, lr):
    config_name = f"{optimizer_name}_{criterion_name}_lr{lr}".replace('.', '_')
    config_dir = os.path.join(model_dir, config_name)
    os.makedirs(config_dir, exist_ok=True)
    log_file = os.path.join(config_dir, "training_log.txt")
    csv_log_path = os.path.join(config_dir, "training_log.csv")

    with open(log_file, "w") as f:
        f.write(f"Start treningu BiLSTM | Optimizer: {optimizer_name} | Criterion: {criterion_name} | LR: {lr}\n")
        f.write("=" * 100 + "\n")

    print(f"Trening: {optimizer_name}, {criterion_name}, LR={lr}")
    print(f"Folder: {config_dir}")

    return config_dir, log_file, csv_log_path


def get_input_dimension(df, col_name):
    if len(df[col_name]) == 0:
        raise ValueError("Brak danych treningowych.")

    if df[col_name].iloc[0].shape[0] == 0:
        for features_array in df[col_name].values:
            if features_array.shape[0] > 0:
                return features_array.shape[1]
        raise ValueError(f"Wszystkie sekwencje {col_name} są puste.")

    return df[col_name].iloc[0].shape[1]


def create_dataloaders(train_df, test_df):
    train_dataset = AudioDataset(train_df)
    test_dataset = AudioDataset(test_df)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=False, collate_fn=collate_fn_padd)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, drop_last=False, collate_fn=collate_fn_padd)
    return train_loader, test_loader


def get_optimizer(model, optimizer_name, lr):
    optimizer_map = {
        "Adam": optim.Adam(model.parameters(), lr=lr),
        "AdamW": optim.AdamW(model.parameters(), lr=lr),
        "SGD": optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    }
    return optimizer_map.get(optimizer_name, optim.Adam(model.parameters(), lr=lr))


def train_single_epoch(model, loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in loader:
        if X_batch.size(0) == 0:
            continue
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    return train_loss / len(loader)


def validate_single_epoch(model, loader, criterion, device):
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for X_val, y_val in loader:
            if X_val.size(0) == 0:
                continue
            X_val, y_val = X_val.to(device), y_val.to(device)
            outputs = model(X_val)
            loss = criterion(outputs, y_val)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y_val.size(0)
            correct += (predicted == y_val).sum().item()

    avg_val_loss = val_loss / len(loader)
    val_accuracy = correct / total if total > 0 else 0
    return avg_val_loss, val_accuracy


def save_plots(df_logs, config_dir, optimizer_name, criterion_name, lr):
    plt.figure(figsize=(8, 5))
    plt.plot(df_logs["epoch"], df_logs["train_loss"], label="Train Loss")
    plt.plot(df_logs["epoch"], df_logs["val_loss"], label="Val Loss")
    plt.title(f"Loss ({optimizer_name}, {criterion_name}, LR={lr})")
    plt.xlabel("Epoki")
    plt.ylabel("Strata")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config_dir, "loss_plot.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df_logs["epoch"], df_logs["val_accuracy"], label="Val Accuracy", color="green")
    plt.title(f"Accuracy ({optimizer_name}, {criterion_name}, LR={lr})")
    plt.xlabel("Epoki")
    plt.ylabel("Dokładność")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config_dir, "accuracy_plot.png"))
    plt.close()


def calculate_final_metrics(model, test_loader, best_model_path, device):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()
    y_true, y_pred, y_scores = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            if outputs.shape[1] == 1:
                probs = torch.sigmoid(outputs).squeeze(1)
                predicted = (probs >= 0.5).float()
                y_scores.extend(probs.cpu().numpy())
            else:
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(probs, 1)
                if probs.shape[1] == 2:
                    y_scores.extend(probs[:, 1].cpu().numpy())
                else:
                    y_scores.extend(torch.max(probs, dim=1)[0].cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())

    y_true, y_pred, y_scores = np.array(y_true), np.array(y_pred), np.array(y_scores)
    f1 = f1_score(y_true, y_pred, average='binary' if len(np.unique(y_true)) == 2 else 'macro')
    acc = accuracy_score(y_true, y_pred)

    eer = None
    if len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        eer = fpr[np.nanargmin(np.abs(fnr - fpr))]
        print(f"Final metrics: F1={f1:.4f}, EER={eer:.4f}, Accuracy={acc:.4f}")
    else:
        print(f"Final metrics: F1={f1:.4f}, Accuracy={acc:.4f}")

    return f1, acc, eer


def BiLSTM_model(
        train_df, test_df, col_name="cqcc", num_epochs=100, model=None,
        criterion_name="CrossEntropyLoss", optimizer_name="Adam", lr=1e-4,
        model_dir="GMM-BiLSTM"):
    config_dir, log_file, csv_log_path = setup_directories(model_dir, optimizer_name, criterion_name, lr)
    train_loader, test_loader = create_dataloaders(train_df, test_df)
    input_dim = get_input_dimension(train_df, col_name)

    if model is None:
        model = BiLSTMClassifier(input_dim=input_dim)

    criterion = nn.CrossEntropyLoss() if criterion_name == "CrossEntropyLoss" else nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, optimizer_name, lr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_model_path = os.path.join(config_dir, "bilstm_best_model.pt")
    worst_model_path = os.path.join(config_dir, "bilstm_worst_model.pt")

    log_data = []
    best_val_loss = float("inf")
    worst_val_loss = float("-inf")

    print(f"Rozpoczęto trening BiLSTM ({optimizer_name}, {criterion_name})...")
    start_time = time.time()

    for epoch in range(num_epochs):
        avg_train_loss = train_single_epoch(model, train_loader, criterion, optimizer, device)
        avg_val_loss, val_accuracy = validate_single_epoch(model, test_loader, criterion, device)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)

        if avg_val_loss > worst_val_loss:
            worst_val_loss = avg_val_loss
            torch.save(model.state_dict(), worst_model_path)

        log_data.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_accuracy": val_accuracy
        })

        log_line = f"Epoch {epoch + 1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_accuracy:.4f}"
        print(log_line)
        with open(log_file, "a") as f:
            f.write(log_line + "\n")

    total_time = time.time() - start_time
    with open(log_file, "a") as f:
        f.write("=" * 100 + "\n")
        f.write(f"Czas treningu: {total_time:.2f} s\n")
        f.write(f"Najlepszy val_loss: {best_val_loss:.4f}\n")
        f.write(f"Najgorszy val_loss: {worst_val_loss:.4f}\n")

    df_logs = pd.DataFrame(log_data)
    df_logs.to_csv(csv_log_path, index=False)
    save_plots(df_logs, config_dir, optimizer_name, criterion_name, lr)

    f1, acc, _ = calculate_final_metrics(model, test_loader, best_model_path, device)

    return model, test_loader, {
        "best_val_loss": best_val_loss,
        "worst_val_loss": worst_val_loss,
        "config_dir": config_dir,
        "f1": f1,
        "accuracy": acc
    }


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