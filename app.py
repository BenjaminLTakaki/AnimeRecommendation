import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)

# Load the dataset
anime = pd.read_csv('anime.csv')

# Preprocess the data
def preprocess_data(anime):
    # Fill missing values
    anime['genre'] = anime['genre'].fillna('')
    anime['type'] = anime['type'].fillna('Unknown')
    anime['rating'] = anime['rating'].fillna(anime['rating'].median())
    anime['members'] = anime['members'].fillna(anime['members'].median())
    anime['episodes'] = anime['episodes'].replace(['Unknown', '?'], 0)
    anime['episodes'] = anime['episodes'].fillna(0)
    anime['episodes'] = anime['episodes'].astype(int)

    # Split genres and create genre list
    anime['genre_list'] = anime['genre'].apply(lambda x: x.split(', '))

    # Encode genres using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    genre_dummies = mlb.fit_transform(anime['genre_list'])
    genre_df = pd.DataFrame(genre_dummies, columns=mlb.classes_)

    # One-hot encode 'type'
    type_ohe = OneHotEncoder()
    type_dummies = type_ohe.fit_transform(anime[['type']]).toarray()
    type_df = pd.DataFrame(type_dummies, columns=type_ohe.get_feature_names_out(['type']))

    # Standardize numerical features
    numerical_features = anime[['rating', 'members', 'episodes']]
    scaler = StandardScaler()
    numerical_scaled = scaler.fit_transform(numerical_features)
    numerical_df = pd.DataFrame(numerical_scaled, columns=['rating', 'members', 'episodes'])

    # Combine all features
    features = pd.concat([genre_df, type_df, numerical_df], axis=1)

    return features, mlb, type_ohe, scaler

# Preprocess data and fit model
features, mlb, type_ohe, scaler = preprocess_data(anime)

# Fit the NearestNeighbors model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(features)

# Recommendation function
def recommend_anime(favorite_anime_list, favorite_genres_list, n_recommendations=10):
    # Get indices of favorite anime
    anime_indices = anime[anime['name'].isin(favorite_anime_list)].index.tolist()

    # Average the features of the favorite anime
    if anime_indices:
        favorite_anime_features = features.iloc[anime_indices].mean(axis=0)
    else:
        favorite_anime_features = pd.Series(np.zeros(features.shape[1]), index=features.columns)

    # Build genre vector for favorite genres
    favorite_genre_vector = np.zeros(len(mlb.classes_))
    for genre in favorite_genres_list:
        if genre in mlb.classes_:
            idx = np.where(mlb.classes_ == genre)[0][0]
            favorite_genre_vector[idx] = 1
    favorite_genre_series = pd.Series(favorite_genre_vector, index=mlb.classes_)

    # Combine favorite genres with other features
    user_profile = favorite_anime_features.copy()
    user_profile.update(favorite_genre_series)

    # Reshape user_profile to 2D array
    user_profile = user_profile.values.reshape(1, -1)

    # Find nearest neighbors
    distances, indices = model.kneighbors(user_profile, n_neighbors=n_recommendations)

    # Get anime names
    recommended_anime = anime.iloc[indices[0]]['name'].values.tolist()

    return recommended_anime

# Flask routes
@app.route('/', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        # Get user input
        favorite_anime_input = request.form.get('favorite_anime', '')
        favorite_genres = request.form.getlist('favorite_genres')

        # Split the favorite anime input into a list
        favorite_anime = [anime.strip() for anime in favorite_anime_input.split(',') if anime.strip()]

        recommendations = recommend_anime(favorite_anime, favorite_genres)
        return render_template('recommendations.html', recommendations=recommendations)
    else:
        # Get list of genres for the form
        genres = sorted(mlb.classes_.tolist())
        return render_template('index.html', genres=genres)

@app.route('/anime_titles')
def anime_titles():
    anime_names = sorted(anime['name'].dropna().unique().tolist())
    return jsonify(anime_names)

if __name__ == '__main__':
    app.run(debug=True)
