import sys
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

# Make project root importable
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.data_prep import load_and_clean_data, get_titles_per_year, get_top_genres
from src.recommender import prepare_features, build_similarity_matrix, get_recommendations
from src.clustering import cluster_titles, cluster_summary


@st.cache_data
def load_data():
    data_path = BASE_DIR / "data" / "titles.csv"
    df = load_and_clean_data(data_path)
    return df


@st.cache_resource
def build_models(df):
    # Prepare recommender features
    df_prep, tfidf, tfidf_matrix = prepare_features(df)
    sim_matrix = build_similarity_matrix(tfidf_matrix)

    # Cluster titles
    df_clustered, kmeans, scaler = cluster_titles(df_prep)

    return df_clustered, sim_matrix, tfidf, kmeans, scaler


def page_overview(df):
    st.subheader("Titles over the Years")
    year_counts = get_titles_per_year(df)
    if year_counts.empty:
        st.write("No release_year information available.")
    else:
        st.bar_chart(year_counts)

    st.subheader("Top Genres")
    top_genres = get_top_genres(df, top_n=10)
    if top_genres.empty:
        st.write("No genre information available.")
    else:
        st.bar_chart(top_genres)

    st.subheader("IMDB Score Distribution")
    if "imdb_score" not in df.columns:
        st.write("No IMDB score data available.")
    else:
        fig, ax = plt.subplots()
        df["imdb_score"].dropna().hist(bins=20, ax=ax)
        ax.set_xlabel("IMDB Score")
        ax.set_ylabel("Count")
        ax.set_title("IMDB Score Distribution")
        st.pyplot(fig)


def page_recommendations(df, sim_matrix):
    st.subheader("Find Similar Titles")

    titles = sorted(df["title"].unique())
    selected_title = st.selectbox("Select a title", options=titles)

    top_k = st.slider("Number of recommendations", min_value=3, max_value=20, value=10)

    if st.button("Recommend"):
        recs = get_recommendations(selected_title, df, sim_matrix, top_k=top_k)
        if recs is None or recs.empty:
            st.write("No recommendations found.")
        else:
            st.write("Recommendations similar to:", f"**{selected_title}**")
            st.dataframe(recs)


def page_clusters(df):
    st.subheader("Title Clusters (by Year / Runtime / Scores)")

    if "cluster" not in df.columns:
        st.write("No cluster information available.")
        return

    # Cluster sizes
    cluster_counts = df["cluster"].value_counts().sort_index()
    st.write("Number of titles in each cluster:")
    st.bar_chart(cluster_counts)

    # Summary table
    st.write("Cluster summary (average scores):")
    summary = cluster_summary(df)
    st.dataframe(summary)

    # Allow user to inspect a cluster
    selected_cluster = st.number_input(
        "View titles in cluster", min_value=int(df["cluster"].min()), max_value=int(df["cluster"].max()), value=int(df["cluster"].min())
    )
    cluster_df = df[df["cluster"] == selected_cluster][
        ["title", "type", "genres_clean", "imdb_score", "release_year"]
    ]
    st.write(f"Sample titles from cluster {selected_cluster}:")
    st.dataframe(cluster_df.head(50))


def main():
    st.title("🎬 Netflix Content Explorer & Recommender")

    df = load_data()
    df_models, sim_matrix, tfidf, kmeans, scaler = build_models(df)

    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Overview", "Recommendations", "Clusters"])

    if page == "Overview":
        page_overview(df_models)
    elif page == "Recommendations":
        page_recommendations(df_models, sim_matrix)
    elif page == "Clusters":
        page_clusters(df_models)


if __name__ == "__main__":
    main()
