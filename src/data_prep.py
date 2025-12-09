import pandas as pd


COLUMNS_TO_KEEP = [
    "title",
    "type",
    "description",
    "release_year",
    "age_certification",
    "runtime",
    "genres",
    "production_countries",
    "imdb_score",
    "imdb_votes",
    "tmdb_popularity",
    "tmdb_score",
]


def clean_genres(g):
    """Convert stringified list of genres into a clean string: 'Drama, Romance'."""
    if pd.isna(g):
        return ""
    g = str(g).strip("[]")
    g = g.replace("'", "").replace('"', "")
    parts = [p.strip() for p in g.split(",")]
    parts = [p for p in parts if p]
    return ", ".join(parts)


def load_and_clean_data(csv_path):
    """
    Load the Netflix titles CSV and perform basic cleaning.
    Returns a cleaned pandas DataFrame.
    """
    df = pd.read_csv(csv_path)

    # Keep only relevant columns (ignore missing columns gracefully)
    existing_cols = [c for c in COLUMNS_TO_KEEP if c in df.columns]
    df = df[existing_cols]

    # Drop rows with no title or description
    for col in ["title", "description"]:
        if col in df.columns:
            df = df.dropna(subset=[col])

    # Handle genres
    if "genres" in df.columns:
        df["genres_clean"] = df["genres"].apply(clean_genres)
    else:
        df["genres_clean"] = ""

    # Bag-of-genres (space-separated tokens)
    df["genres_bag"] = df["genres_clean"].str.replace(", ", " ", regex=False)

    return df


def get_titles_per_year(df):
    """Return a Series: release_year -> count of titles (for charts)."""
    if "release_year" not in df.columns:
        return pd.Series(dtype=int)
    return df["release_year"].value_counts().sort_index()


def get_top_genres(df, top_n=10):
    """Return a Series of top N genres and their counts."""
    from collections import Counter

    genre_counter = Counter()
    for g in df["genres_clean"]:
        if g:
            genre_counter.update(g.split(", "))

    if not genre_counter:
        return pd.Series(dtype=int)

    genres, counts = zip(*genre_counter.most_common(top_n))
    return pd.Series(counts, index=genres, name="count")
