import streamlit as st
import pickle
import pandas as pd

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")
st.title("🎥 Movie Recommendation System")
st.write("Discover movies similar to your favorites!")

movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

def recommend(movie):
    index = movies[movies['title'].str.lower() == movie.lower()].index
    if len(index) == 0:
        return ["Movie not found! Please check the name."]
    index = index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movies = []
    for i in distances[1:6]:
        recommended_movies.append(movies.iloc[i[0]].title)
    return recommended_movies

selected_movie = st.selectbox('🎬 Choose a movie:', movies['title'].values)

if st.button('Recommend'):
    recommendations = recommend(selected_movie)
    st.subheader('✨ Recommended Movies:')
    for name in recommendations:
        st.write(f"- {name}")