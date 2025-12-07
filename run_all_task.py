import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, f1_score, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

from ASV_dl_func import add_dataAugmentation, extract_features, balance_func, train_all_features, filtr_nan, \
    transpose_cqcc, prepare_train_test_data
from classic_ml_func import model_gridsearch
from gmm_bilstm_pipeline import BiLSTM_model, create_dataloaders, evaluate_all_models
from prepare_data import prepare_dataframe, prepare_dirs_dataset, balance_class_distribution, extract_cqcc, \
    extract_gtcc, extract_mel_spectrogram, extract_mfcc, extract_lfcc


def get_mixed_dataset(inthewild_path, ASV_metadata: str, ASV_flac_folder: list):

    all_data = {
        "la_df": {
            "metadata": ASV_metadata,
            "flac": ASV_flac_folder,
        }
    }
    train_df_ASV, val_df_ASV = prepare_dataframe(all_data)


    dfs = prepare_dirs_dataset(inthewild_path)
    train_df_inthewild = dfs[2]
    val_df_inthewild = dfs[1]
    test_df_inthewild = dfs[0]

    train_df_inthewild['label'] = train_df_inthewild['label'].map({"fake": 1, "real": 0})
    val_df_inthewild['label'] = val_df_inthewild['label'].map({"fake": 1, "real": 0})
    test_df_inthewild['label'] = test_df_inthewild['label'].map({"fake": 1, "real": 0})

    train_df_inthewild = train_df_inthewild[['file_path', 'label']]
    val_df_inthewild = val_df_inthewild[['file_path', 'label']]
    test_df_inthewild = test_df_inthewild[['file_path', 'label']]

    train_df_ASV = train_df_ASV[['file_path', 'label']]
    val_df_ASV = val_df_ASV[['file_path', 'label']]

    train_df = pd.concat([train_df_inthewild, train_df_ASV], ignore_index=True)
    val_df = pd.concat([val_df_inthewild, val_df_ASV], ignore_index=True)

    balanced_train_df = balance_class_distribution(train_df)
    balanced_val_df = balance_class_distribution(val_df)

    return balanced_train_df, balanced_val_df, test_df_inthewild

def train_pipeline(epochs=100, target_name='label', in_the_wild_dir=None, la_metadata=None, la_flac_folders=None, feature_extractors_map=None):

    if feature_extractors_map is None:
        feature_extractors_map = {
            'cqcc': extract_cqcc,
            'gtcc': extract_gtcc,
            'mel-spect': extract_mel_spectrogram,
            'mfcc': extract_mfcc,
            'lfcc': extract_lfcc
        }

    feature_name = feature_extractors_map.keys()

    train_df, val_df, test_df = get_mixed_dataset(in_the_wild_dir, la_metadata, la_flac_folders)
    train_aug = add_dataAugmentation(train_df)

    train_df_prepared = extract_features(train_aug, feature_extractors_map)
    train_df_prepared = train_df_prepared.dropna(subset=feature_name)

    val_df = extract_features(val_df, feature_extractors_map)
    val_df_prepared = val_df.dropna(subset=feature_name)

    test_df = extract_features(test_df, feature_extractors_map)
    test_df_prepared = test_df.dropna(subset=feature_name)

    train_df_noscale = balance_func(train_df_prepared, col_name=target_name)
    val_df_noscale = balance_func(val_df_prepared, col_name=target_name)
    test_df_noscale = balance_func(test_df_prepared, col_name=target_name)

    # ResNet
    trained_models = train_all_features(train_df_noscale, feature_name, epochs=epochs, label_col=target_name,
                                        test_df=val_df_noscale)

    # BiLSTM
    cqcc_map = {"cqcc": extract_cqcc}

    train_df_noscale = extract_features(train_aug, cqcc_map)
    val_df_noscale = extract_features(val_df, cqcc_map)

    train_bilstm = filtr_nan(train_df_noscale)
    train_bilstm['cqcc'] = train_bilstm['cqcc'].apply(transpose_cqcc)

    test_bilstm = filtr_nan(val_df_noscale)
    test_bilstm['cqcc'] = test_bilstm['cqcc'].apply(transpose_cqcc)

    final_df = train_bilstm[train_bilstm['cqcc'].notnull()]
    final_df_balanced = balance_func(final_df, col_name='label')

    test_df = test_bilstm[test_bilstm['cqcc'].notnull()]
    test_df_balanced = balance_func(test_df, col_name='label')

    train_df, test_df, scaler = prepare_train_test_data(final_df_balanced, test_df=test_df_balanced, label_name='label')

    all_results, test_loader = BiLSTM_model(train_df, test_df, num_epochs=epochs)
    train_loader, val_loader = create_dataloaders(train_df, test_df)
    evaluate_all_models("GMM-BiLSTM", train_loader=train_loader, train_df=train_df, test_df=test_df,
                        test_loader=val_loader)

    # GridSearch Ml models

    model_gridsearch(train_df_noscale, test_df_noscale, feature_name)

def evaluate_test_metrics(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    fpr, tpr, thr = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = max(fpr[idx], fnr[idx])
    auc = roc_auc_score(y_true, y_score)
    y_pred = (y_score >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    return {
        "roc_auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thr,
        "confusion_matrix": cm,
        "f1": f1,
        "accuracy": acc,
        "recall": rec,
        "precision": prec,
        "eer": eer
    }

def visualize_roc_confusion(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    y_pred = (y_score >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(fpr, tpr, label=f"AUC={auc:.4f}")
    axes[0].plot([0, 1], [0, 1], linestyle='--', color='gray')
    axes[0].set_xlabel('FPR')
    axes[0].set_ylabel('TPR')
    axes[0].set_title('ROC Curve')
    axes[0].legend(loc='lower right')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axes[1])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('True')
    axes[1].set_title('Confusion Matrix')
    plt.tight_layout()
    plt.show()