import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


NUMERIC_FEATURES = ["release_year", "runtime", "imdb_score", "tmdb_score"]


def cluster_titles(df: pd.DataFrame, n_clusters=8, random_state=42):
    """
    Cluster titles using year, runtime, imdb_score, tmdb_score.
    Returns: (df_with_cluster, kmeans_model, scaler)
    """
    df = df.copy()

    # Keep only existing numeric columns
    features = [c for c in NUMERIC_FEATURES if c in df.columns]
    if not features:
        raise ValueError("No numeric features available for clustering.")

    feat = df[features].fillna(0)

    scaler = StandardScaler()
    X = scaler.fit_transform(feat)

    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    clusters = kmeans.fit_predict(X)

    df["cluster"] = clusters
    return df, kmeans, scaler


def cluster_summary(df: pd.DataFrame):
    """
    Return a simple summary per cluster: count and average imdb_score / tmdb_score.
    """
    agg_cols = {}
    if "imdb_score" in df.columns:
        agg_cols["imdb_score"] = "mean"
    if "tmdb_score" in df.columns:
        agg_cols["tmdb_score"] = "mean"

    if not agg_cols:
        agg_cols = None

    return df.groupby("cluster").agg(agg_cols).assign(size=df.groupby("cluster").size())
