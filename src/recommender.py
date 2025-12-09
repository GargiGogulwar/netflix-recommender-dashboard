import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def prepare_features(df: pd.DataFrame):
    """
    Combine genres + description into a text field and build a TF-IDF matrix.
    Returns: (df_reset, tfidf_vectorizer, tfidf_matrix)
    """
    df = df.reset_index(drop=True).copy()

    # Make sure columns exist
    if "description" not in df.columns:
        df["description"] = ""
    if "genres_bag" not in df.columns:
        df["genres_bag"] = ""

    df["description"] = df["description"].fillna("")
    df["genres_bag"] = df["genres_bag"].fillna("")

    # Combine text features
    df["text"] = df["genres_bag"] + " " + df["description"]

    tfidf = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df["text"])

    return df, tfidf, tfidf_matrix


def build_similarity_matrix(tfidf_matrix):
    """Compute cosine similarity matrix from TF-IDF matrix."""
    sim_matrix = cosine_similarity(tfidf_matrix)
    return sim_matrix


def get_recommendations(title, df, sim_matrix, top_k=10):
    """
    Given a title string, return a DataFrame of top_k similar titles.
    """
    if "title" not in df.columns:
        raise ValueError("DataFrame must have a 'title' column.")

    title_lower = title.lower()

    # Exact match
    matches = df[df["title"].str.lower() == title_lower]

    # Fallback: partial match
    if matches.empty:
        matches = df[df["title"].str.lower().str.contains(title_lower)]

    if matches.empty:
        return None

    # Take the first matching index
    idx = matches.index[0]

    scores = list(enumerate(sim_matrix[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    # Skip the first one (itself)
    top_indices = [i for i, _ in scores[1 : top_k + 1]]

    cols_to_show = [
        c
        for c in ["title", "type", "genres_clean", "imdb_score", "release_year"]
        if c in df.columns
    ]
    recs = df.iloc[top_indices][cols_to_show].copy()

    return recs
