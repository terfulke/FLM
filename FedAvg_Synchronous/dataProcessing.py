import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    # 1) Load & drop NaNs
    df = pd.read_csv(csv_path)
    df.dropna(inplace=True)

    # 2) Drop pure-ID columns
    df.drop(columns=["ID", "Customer_ID", "Name"], inplace=True, errors="ignore")

    # 3) Encode any remaining object-dtype column
    for col in df.select_dtypes(include=["object"]):
        df[col] = df[col].astype(str)
        df[col], _ = pd.factorize(df[col])

    # 4) Move target out and scale features
    df["target"] = df["Credit_Score"]
    df.drop(columns=["Credit_Score"], inplace=True)
    scaler = StandardScaler()
    feats = df.drop(columns=["target"])
    df_scaled = pd.DataFrame(scaler.fit_transform(feats), columns=feats.columns)
    df_scaled["target"] = df["target"].values

    return df_scaled


def make_client_splits(
    df: pd.DataFrame,
    num_clients: int,
    strategy: str = "iid",
) -> list[pd.DataFrame]:
    """
    Returns a list of `num_clients` DataFrames,
    split either IID or non-IID by `strategy`.
    """
    if strategy == "iid":
        # Simple random shuffle + equal split
        return np.array_split(df.sample(frac=1, random_state=42), num_clients)

    elif strategy == "noniid":
        # Group by label, shard each group, then interleave shards
        groups = [g for _, g in df.groupby("target")]
        shards = [np.array_split(g, num_clients) for g in groups]

        clients: list[pd.DataFrame] = []
        for i in range(num_clients):
            parts = []
            for shard in shards:
                if i < len(shard):
                    parts.append(shard[i])
                else:
                    # If some group was smaller, pad with empty
                    parts.append(pd.DataFrame(columns=df.columns))
            clients.append(pd.concat(parts, ignore_index=True))
        return clients

    else:
        raise ValueError(f"Unknown split strategy: {strategy}")
