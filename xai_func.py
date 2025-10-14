from sklearn.preprocessing import StandardScaler
import pandas as pd

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