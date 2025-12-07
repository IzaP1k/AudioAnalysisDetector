import os
import json
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import xgboost as xgb



def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    return eer


def evaluate_performance(model, X_test, y_test, grid_search_results=None):
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "eer": compute_eer(y_test, y_score)
    }

    if grid_search_results:
        metrics["best_params"] = grid_search_results.best_params_
        metrics["best_cv_score"] = grid_search_results.best_score_

    return metrics



def row_stats(X):
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1, ddof=1)
    return np.concatenate([mean, std], axis=1)


def prepare_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def extract_and_process_data(df_train, df_test, feature, label):
    X_train = np.array(df_train[feature].tolist())
    X_test = np.array(df_test[feature].tolist())

    y_train = df_train[label].values
    y_test = df_test[label].values

    X_train_stats = row_stats(X_train)
    X_test_stats = row_stats(X_test)

    X_train_final, X_test_final, scaler = prepare_features(X_train_stats, X_test_stats)

    return X_train_final, X_test_final, y_train, y_test, scaler



def save_artifacts(base_dir, model_name, feature_name, model, scaler, metrics):
    output_dir = os.path.join(base_dir, model_name, feature_name)
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "model.pkl"), "wb") as f:
        pickle.dump(model, f)

    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Saved results for {model_name} on {feature_name}")



def run_grid_search(estimator, param_grid, X_train, y_train):
    grid_search = GridSearchCV(
        estimator,
        param_grid,
        cv=5,
        scoring='f1',
        verbose=3,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    return grid_search


def xgboost_gridsearch(X_train, X_test, y_train, y_test, param_grid=None):
    if param_grid is None:
        param_grid = {
            'max_depth': [3, 5, 7],
            'learning_rate': [0.1, 0.01],
            'subsample': [0.7, 1.0]
        }

    gs = run_grid_search(xgb.XGBClassifier(), param_grid, X_train, y_train)
    best_model = gs.best_estimator_
    metrics = evaluate_performance(best_model, X_test, y_test, gs)

    return best_model, metrics


def svm_gridsearch(X_train, X_test, y_train, y_test, param_grid=None):
    if param_grid is None:
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 0.01, 0.001],
            'kernel': ['rbf']
        }

    gs = run_grid_search(SVC(probability=True), param_grid, X_train, y_train)
    best_model = gs.best_estimator_
    metrics = evaluate_performance(best_model, X_test, y_test, gs)

    return best_model, metrics



def model_gridsearch(df_train, df_test, feature_list, label='label'):
    base_dir = "gridSearch"
    os.makedirs(base_dir, exist_ok=True)

    for feature in feature_list:
        print(f"\n=== PROCESSING FEATURE: {feature} ===")

        X_train, X_test, y_train, y_test, scaler = extract_and_process_data(
            df_train, df_test, feature, label
        )

        svm_model, svm_metrics = svm_gridsearch(X_train, X_test, y_train, y_test)
        save_artifacts(base_dir, "svm", feature, svm_model, scaler, svm_metrics)

        xgb_model, xgb_metrics = xgboost_gridsearch(X_train, X_test, y_train, y_test)
        save_artifacts(base_dir, "xgboost", feature, xgb_model, scaler, xgb_metrics)