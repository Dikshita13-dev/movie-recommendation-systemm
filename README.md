# 🎬 Movie Recommendation System

A content-based movie recommendation system built with **Python** and **Streamlit** that recommends similar movies based on genre similarity. The application uses the **MovieLens ml-latest-small** dataset for recommendations and integrates the **TMDB API** to display movie posters.

## Features

* 🎥 Browse movies from the MovieLens dataset
* 🤖 Content-based recommendations using TF-IDF and Cosine Similarity
* 🖼️ Movie posters fetched from TMDB API
* 🎛️ Interactive Streamlit interface
* ⚡ Fast recommendations using cached computations
* 📊 Adjustable number of recommendations

## Tech Stack

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-learn
* Requests
* MovieLens Dataset
* TMDB API

## Dataset

This project uses the **MovieLens Latest Small** dataset, which contains movie metadata and links to TMDB IDs.

Download the dataset from:
https://grouplens.org/datasets/movielens/latest/

After downloading, place the extracted folder in the project directory:

```text
movie-recommender/
│── app.py
│── requirements.txt
│── README.md
└── ml-latest-small/
    ├── movies.csv
    ├── links.csv
    ├── ratings.csv
    ├── tags.csv
```

## How It Works

1. Load movie information from the MovieLens dataset.
2. Clean and preprocess movie genres.
3. Convert genres into TF-IDF vectors.
4. Compute cosine similarity between all movies.
5. Select a movie from the dropdown.
6. Recommend the most similar movies.
7. Fetch movie posters from the TMDB API.

## Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<repository-name>.git
cd <repository-name>
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Requirements

```text
streamlit
pandas
numpy
scikit-learn
requests
```

## TMDB API Key

This project uses the TMDB API only for fetching movie posters.

Create a free API key from:
https://www.themoviedb.org/settings/api

Enter the API key in the sidebar after launching the application.

## Run the Project

```bash
streamlit run app.py
```

## Screenshots

Add screenshots of the application here after deployment.

## Future Improvements

* Recommend movies using plot summaries in addition to genres
* Hybrid recommendation system (content + collaborative filtering)
* Search by actor or director
* Movie trailers
* User ratings and favorites
* Genre filtering
* Deploy on Streamlit Community Cloud

