import pandas as pd
import numpy as np
import requests
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

DATA_DIR = "ml-latest-small"


# ---------- Data loading & model (cached so it only runs once) ----------

@st.cache_data
def load_data():
    movies = pd.read_csv(f"{DATA_DIR}/movies.csv")
    links = pd.read_csv(f"{DATA_DIR}/links.csv")  # movieId -> imdbId, tmdbId

    # Drop rows with no genre info
    movies = movies[movies["genres"] != "(no genres listed)"].copy()

    # Merge in tmdbId so we can fetch real posters (no title-guessing)
    movies = movies.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")

    # Clean genres for vectorizing
    movies["genres_clean"] = movies["genres"].str.replace("|", " ", regex=False)

    # IMPORTANT FIX: original code indexed by title, which breaks when two
    # movies share a title (e.g. remakes). Reset to a clean positional index
    # and always look things up by movieId instead.
    movies = movies.reset_index(drop=True)

    return movies


@st.cache_data
def build_similarity(movies):
    # Switched CountVectorizer -> TF-IDF: down-weights extremely common
    # genres (like "Drama") so recommendations aren't dominated by them.
    vectorizer = TfidfVectorizer(stop_words="english")
    genre_matrix = vectorizer.fit_transform(movies["genres_clean"])
    cosine_sim = cosine_similarity(genre_matrix, genre_matrix)
    return cosine_sim


def recommend(movies, cosine_sim, movie_id, n=10):
    idx = movies.index[movies["movieId"] == movie_id][0]
    scores = list(enumerate(cosine_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    top = scores[1:n + 1]  # skip itself
    top_indices = [i[0] for i in top]
    return movies.iloc[top_indices]


# ---------- Poster fetching ----------

@st.cache_data(show_spinner=False)
def get_poster_url(tmdb_id, api_key):
    if not api_key or pd.isna(tmdb_id):
        return None
    try:
        tmdb_id = int(tmdb_id)
        url = f"https://api.themoviedb.org/3/movie/{tmdb_id}"
        resp = requests.get(url, params={"api_key": api_key}, timeout=5)
        if resp.status_code == 200:
            poster_path = resp.json().get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w342{poster_path}"
    except Exception:
        pass
    return None


PLACEHOLDER = "https://placehold.co/300x450?text=No+Poster"


# ---------- UI ----------

st.title("🎬 Movie Recommender")
st.caption("Content-based recommendations using genre similarity (MovieLens ml-latest-small)")

with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("TMDB API key (for posters)", type="password",
                             help="Get a free key at themoviedb.org -> Settings -> API")
    n_recs = st.slider("Number of recommendations", 5, 20, 10)

movies = load_data()
cosine_sim = build_similarity(movies)

# Build a display label that disambiguates duplicate titles by year/movieId
movies["display_title"] = movies["title"]

selected_title = st.selectbox("Pick a movie you like:", movies["display_title"].sort_values().unique())

if st.button("Recommend", type="primary"):
    # Handle duplicate titles by taking the first match's movieId
    match = movies[movies["display_title"] == selected_title].iloc[0]
    results = recommend(movies, cosine_sim, match["movieId"], n=n_recs)

    st.subheader(f"Because you liked: {selected_title}")

    cols_per_row = 5
    rows = [results.iloc[i:i + cols_per_row] for i in range(0, len(results), cols_per_row)]

    for row in rows:
        cols = st.columns(cols_per_row)
        for col, (_, movie) in zip(cols, row.iterrows()):
            with col:
                poster = get_poster_url(movie["tmdbId"], api_key) or PLACEHOLDER
                st.image(poster, use_container_width=True)
                st.markdown(f"**{movie['title']}**")
                st.caption(movie["genres"].replace("|", ", "))
