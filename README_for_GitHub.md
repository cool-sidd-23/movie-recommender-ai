# 🎥 Movie Recommendation System

An AI-powered movie recommendation system that suggests similar movies based on user preferences using **content-based filtering**.  
Built with Python and machine learning techniques to enhance user experience by analyzing movie similarities.

---

## 🚀 Features

- 🎯 Content-based recommendations using **cosine similarity**
- 📊 Movie metadata analysis (genre, overview, keywords, etc.)
- 💻 Interactive web app built with **Streamlit**
- 🔍 Search for any movie and get AI-based recommendations
- ⚙️ Simple and efficient — uses no external API

---

## 🧠 Tech Stack

- **Python 3.x**  
- **Pandas**, **NumPy** – data processing  
- **Scikit-learn** – cosine similarity, TF-IDF vectorization  
- **Streamlit** – web app interface  
- **Pickle** – model serialization  

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/cool-sidd-23/movie-recommender-ai.git
   cd movie-recommender-ai
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**
   ```bash
   streamlit run app.py
   ```

---

## 🧩 Project Structure

```
movie-recommender-ai/
│
├── app.py                # Streamlit frontend
├── model.py              # Core recommendation logic
├── movies.csv            # Movie dataset (TMDb or IMDb)
├── similarity.pkl        # Precomputed similarity matrix
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## 💡 How It Works

1. **Data Preprocessing**  
   The dataset is cleaned and relevant columns like `title`, `overview`, `genres`, and `keywords` are merged into a single text field.  

2. **Feature Extraction**  
   TF-IDF or Count Vectorizer is applied to convert text data into numerical vectors.

3. **Similarity Calculation**  
   Cosine similarity is computed between all movie vectors to identify similar ones.

4. **Recommendation**  
   When a user selects a movie, the system finds the top N most similar movies.

---

## 🧾 Sample Code

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
movies = pd.read_csv('movies.csv')
movies['tags'] = movies['overview'] + " " + movies['genres'] + " " + movies['keywords']

# Convert text to vectors
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

# Compute similarity
similarity = cosine_similarity(vectors)

def recommend(movie):
    index = movies[movies['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    for i in distances[1:6]:
        print(movies.iloc[i[0]].title)
```

---

## 🌐 Streamlit Frontend Example

```python
import streamlit as st
import pickle
import pandas as pd

st.title('🎬 Movie Recommender System')

movies_dict = pickle.load(open('movie_dict.pkl', 'rb'))
movies = pd.DataFrame(movies_dict)
similarity = pickle.load(open('similarity.pkl', 'rb'))

selected_movie = st.selectbox('Select a movie:', movies['title'].values)

if st.button('Recommend'):
    names = recommend(selected_movie)
    for name in names:
        st.write(name)
```

---

## 🧰 Requirements

```txt
pandas
numpy
scikit-learn
streamlit
pickle-mixin
```

---

## 🚀 Deployment

You can deploy the Streamlit app for free on:  
- [Streamlit Cloud](https://streamlit.io/cloud)  
- [Hugging Face Spaces](https://huggingface.co/spaces)

---

## 🌟 Future Enhancements

- 🎬 Integrate **TMDb API** for posters and movie info  
- 👥 Add collaborative filtering  
- 📱 Responsive UI with posters and ratings  
- 🧠 Fine-tune with deep learning (e.g., Autoencoders)

---

## 🧑‍💻 Author

**Siddhartha Gupta**  
🔗 [GitHub](https://github.com/cool-sidd-23) | [LinkedIn](https://www.linkedin.com/in/siddhartha-gupta-789368191/)
