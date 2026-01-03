import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

movies = pd.read_csv("/Users/komalkumari/Downloads/ml-latest-small/movies.csv")


movies["genres_clean"] = movies["genres"].str.replace("|", " ", regex = False)
movies[["title", "genres_clean"]].head()
movies = movies[movies["genres"] != "(no genres listed)"]

vectorizer = CountVectorizer(stop_words = "english")
genre_matrix = vectorizer.fit_transform(movies["genres_clean"])
cosine_sim = cosine_similarity(genre_matrix, genre_matrix)
movies_indices = pd.Series(movies.index, index = movies["title"])

def recommended_movies(title, cosine_sim = cosine_sim, n =10):
  idx = movies_indices[title]
  similarity_scores = list(enumerate(cosine_sim[idx]))
  similarity_scores = sorted(similarity_scores, key = lambda x : x[1], reverse = True)
  top_movies = similarity_scores[1:n+1]
  movies_indices_list = [i[0] for i in top_movies]
  return movies['title'].iloc[movies_indices_list]
recommended_movies("Jumanji (1995)")