# movie_recommender_svm.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import requests
from io import StringIO

st.title("🎬 SVM-Powered Movie Recommender")
st.write("Rate a few movies, and I'll use Support Vector Machines to predict what you'll love!")

# Load MovieLens 100k dataset (small but perfect for demo)
@st.cache_data
def load_data():
    # Movies
    movies_url = "https://files.grouplens.org/datasets/movielens/ml-100k/u.item"
    movies = pd.read_csv(movies_url, sep='|', encoding='latin-1', header=None, usecols=[0,1,2])
    movies.columns = ['movie_id', 'title', 'year']
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
    movies['title'] = movies['title'].str.replace(r'\(\d{4}\)', '', regex=True).str.strip()
    
    # Genres
    genre_cols = ['unknown', 'Action', 'Adventure', 'Animation', 'Children', 'Comedy',
                  'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror',
                  'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
    genres = pd.read_csv(movies_url, sep='|', encoding='latin-1', header=None, usecols=range(5,24))
    genres.columns = genre_cols[1:]
    movies = pd.concat([movies, genres], axis=1)
    
    return movies

movies = load_data()

st.write(f"Loaded {len(movies)} movies!")

# User rates some movies
st.subheader("Rate 10 movies to train your personal SVM model")
sample_movies = movies.sample(10, random_state=42)

user_ratings = {}
for _, row in sample_movies.iterrows():
    rating = st.slider(
        f"{row['title']} ({row['year']})",
        min_value=1, max_value=5, value=3, step=1,
        key=row['movie_id']
    )
    user_ratings[row['movie_id']] = rating

if st.button("🍿 Train SVM & Get Recommendations"):
    with st.spinner("Training your personal SVM model..."):
        # Prepare training data from user ratings
        rated_movie_ids = list(user_ratings.keys())
        X_user = []
        y_user = []
        
        for mid in rated_movie_ids:
            movie_row = movies[movies['movie_id'] == mid].iloc[0]
            # Features: year + one-hot genres
            features = [int(movie_row['year'] or 1995)]
            features.extend(movie_row[4:].values)  # genre columns
            X_user.append(features)
            
            # Label: 1 if liked (≥4), 0 otherwise
            y_user.append(1 if user_ratings[mid] >= 4 else 0)
        
        X_user = np.array(X_user)
        y_user = np.array(y_user)
        
        # Train SVM
        svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
        svm.fit(X_user, y_user)
        
        # Predict on all movies
        X_all = []
        movie_indices = []
        for idx, row in movies.iterrows():
            features = [int(row['year'] or 1995)]
            features.extend(row[4:].values)
            X_all.append(features)
            movie_indices.append(idx)
        
        X_all = np.array(X_all)
        probabilities = svm.predict_proba(X_all)[:, 1]  # probability of liking
        
        # Get top recommendations (not already rated)
        rec_df = pd.DataFrame({
            'movie_id': movies['movie_id'],
            'title': movies['title'],
            'year': movies['year'],
            'like_prob': probabilities,
            'idx': movie_indices
        })
        rec_df = rec_df[~rec_df['movie_id'].isin(rated_movie_ids)]
        recommendations = rec_df.sort_values('like_prob', ascending=False).head(10)
        
        st.success("SVM Model Trained Successfully!")
        st.write(f"Accuracy on your ratings: {accuracy_score(y_user, svm.predict(X_user)):.2f}")
        
        st.subheader("🎯 Top 10 Movies You'll Probably LOVE")
        for _, rec in recommendations.iterrows():
            st.write(f"**{rec['title']}** ({rec['year']}) – {rec['like_prob']:.2%} chance you'll rate it 4+ stars")

st.write("\nBuilt with ❤️ using Support Vector Machines + Streamlit")