import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import numpy as np

def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)
    # Drop ID columns
    df.drop(columns=["ID", "Customer_ID", "Name"], inplace=True, errors="ignore")
    # Encode all object cols
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].astype(str)
        df[col], _ = pd.factorize(df[col])
    # Separate target
    df['target'] = df['Credit_Score']
    df.drop(columns=['Credit_Score'], inplace=True)
    # Scale features
    scaler = StandardScaler()
    features = df.drop(columns=['target'])
    df_scaled = pd.DataFrame(scaler.fit_transform(features), columns=features.columns)
    df_scaled['target'] = df['target'].values
    return df_scaled


def make_client_splits(df: pd.DataFrame, num_clients: int):
    # Shuffle and split
    return np.array_split(df.sample(frac=1, random_state=42), num_clients)
