import pandas as pd
import numpy as np
from flask import Flask, request, render_template, jsonify, redirect, url_for, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import sqlite3
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.neighbors import NearestNeighbors

app = Flask(__name__)
app.secret_key = 'your_secret_key'

login_manager = LoginManager()
login_manager.init_app(app)

# User model
class User(UserMixin):
    def __init__(self, id, username, password):
        self.id = id
        self.username = username
        self.password = password

# Load the dataset
anime = pd.read_csv('anime.csv')

# Preprocess the data
def preprocess_data(anime):
    # Filter to include only base anime (e.g., type is 'TV')
    base_anime = anime[anime['type'] == 'TV'].copy()

    # Fill missing values
    base_anime['genre'] = base_anime['genre'].fillna('')
    base_anime['type'] = base_anime['type'].fillna('Unknown')
    base_anime['rating'] = base_anime['rating'].fillna(base_anime['rating'].median())
    base_anime['members'] = base_anime['members'].fillna(base_anime['members'].median())
    base_anime['episodes'] = base_anime['episodes'].replace(['Unknown', '?'], 0)
    base_anime['episodes'] = base_anime['episodes'].fillna(0).astype(int)

    # Split genres and create genre list
    base_anime['genre_list'] = base_anime['genre'].apply(lambda x: x.split(', '))

    # Remove empty genres
    base_anime = base_anime[base_anime['genre'].str.strip() != '']

    # Encode genres using MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    genre_dummies = mlb.fit_transform(base_anime['genre_list'])
    genre_df = pd.DataFrame(genre_dummies, columns=mlb.classes_)

    # One-hot encode 'type'
    type_ohe = OneHotEncoder()
    type_dummies = type_ohe.fit_transform(base_anime[['type']]).toarray()
    type_df = pd.DataFrame(type_dummies, columns=type_ohe.get_feature_names_out(['type']))

    # Standardize numerical features
    numerical_features = base_anime[['rating', 'members', 'episodes']]
    scaler = StandardScaler()
    numerical_scaled = scaler.fit_transform(numerical_features)
    numerical_df = pd.DataFrame(numerical_scaled, columns=['rating', 'members', 'episodes'])

    # Combine all features
    features = pd.concat([genre_df, type_df, numerical_df], axis=1)

    return features, mlb, type_ohe, scaler, base_anime

# Preprocess data and fit model
features, mlb, type_ohe, scaler, base_anime = preprocess_data(anime)

# Fit the NearestNeighbors model
model = NearestNeighbors(metric='cosine', algorithm='brute')
model.fit(features)

def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    # Create saved_anime table
    c.execute('''
        CREATE TABLE IF NOT EXISTS saved_anime (
            user_id INTEGER,
            anime_name TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')
    conn.commit()
    conn.close()

init_db()

@login_manager.user_loader
def load_user(user_id):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('SELECT id, username, password FROM users WHERE id = ?', (user_id,))
    user = c.fetchone()
    conn.close()
    if user:
        return User(id=user[0], username=user[1], password=user[2])
    return None

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        try:
            c.execute('INSERT INTO users (username, password) VALUES (?, ?)', (username, password))
            conn.commit()
            user = User(id=c.lastrowid, username=username, password=password)
            login_user(user)
            return redirect(url_for('recommend'))
        except sqlite3.IntegrityError:
            return 'Username already exists'
        finally:
            conn.close()
    else:
        return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('SELECT id, username, password FROM users WHERE username = ?', (username,))
        user = c.fetchone()
        conn.close()
        if user and user[2] == password:
            user_obj = User(id=user[0], username=user[1], password=user[2])
            login_user(user_obj)
            return redirect(url_for('recommend'))
        else:
            return 'Invalid credentials'
    else:
        return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/save_recommendation', methods=['POST'])
@login_required
def save_recommendation():
    anime = request.form['anime']
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('INSERT INTO saved_anime (user_id, anime_name) VALUES (?, ?)', (current_user.id, anime))
    conn.commit()
    conn.close()
    return jsonify({'status': 'success'})

@app.route('/get_saved_recommendations', methods=['GET'])
@login_required
def get_saved_recommendations():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('SELECT anime_name FROM saved_anime WHERE user_id = ?', (current_user.id,))
    saved = [row[0] for row in c.fetchall()]
    conn.close()
    return jsonify({'saved': saved})

@app.route('/saved_recommendations')
@login_required
def saved_recommendations():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('SELECT anime_name FROM saved_anime WHERE user_id = ?', (current_user.id,))
    saved = [row[0] for row in c.fetchall()]
    conn.close()
    return render_template('saved_recommendations.html', saved=saved)

# Recommendation function
def recommend_anime(favorite_anime_list, favorite_genres_list, n_recommendations=10):
    # Get indices of favorite anime from base_anime
    anime_indices = base_anime[base_anime['name'].isin(favorite_anime_list)].index.tolist()

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

    # Get anime names from base_anime
    recommended_anime = base_anime.iloc[indices[0]]['name'].values.tolist()

    if current_user.is_authenticated:
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute('SELECT anime_name FROM saved_anime WHERE user_id = ?', (current_user.id,))
        saved_anime = set([row[0] for row in c.fetchall()])
        conn.close()
        recommended_anime = [anime for anime in recommended_anime if anime not in saved_anime]
        recommended_anime = recommended_anime[:n_recommendations]

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
    # Return anime names from base_anime
    anime_names = sorted(base_anime['name'].dropna().unique().tolist())
    return jsonify(anime_names)

if __name__ == '__main__':
    app.run(debug=True)