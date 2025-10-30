import pandas as pd
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("📦 Loading dataset...")

movies = pd.read_csv('tmdb_5000_movies.csv')
movies = movies[['id', 'title', 'overview', 'genres', 'keywords']]
movies.dropna(inplace=True)

import ast
def convert(text):
    try:
        return [i['name'] for i in ast.literal_eval(text)]
    except:
        return []

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords']
new_df = movies[['id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

similarity = cosine_similarity(vectors)

pickle.dump(new_df, open('movie_dict.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

print("✅ Model built successfully and files saved!")