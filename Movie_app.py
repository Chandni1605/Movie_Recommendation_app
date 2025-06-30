import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the data
@st.cache_data
def load_data():
    df = pd.read_csv("/Users/chandnisingh/Desktop/Google classroom notes/4th/MOVIE RATINGS _ ADVANCE VISUALIZATION _ EDA 1/Movie-Rating.csv")
    df.columns = ['Film', 'Genre', 'RT_Rating', 'Audience_Rating', 'Budget_Million', 'Year']
    df = df.dropna()
    return df

df = load_data()
df['Film'] = df['Film'].str.strip()
df['Genre'] = df['Genre'].fillna('')

# Process features
tfidf = TfidfVectorizer()
genre_matrix = tfidf.fit_transform(df['Genre'])

numeric_features = df[['RT_Rating', 'Audience_Rating', 'Budget_Million']]
scaler = StandardScaler()
scaled_numeric = scaler.fit_transform(numeric_features)

combined_features = np.hstack((genre_matrix.toarray(), scaled_numeric))
similarity_matrix = cosine_similarity(combined_features)
indices = pd.Series(df.index, index=df['Film']).drop_duplicates()

# Recommender function
def get_recommendations(title, num=5):
    title = title.strip()
    if title not in indices:
        return []
    idx = indices[title]
    sim_scores = list(enumerate(similarity_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num+1]
    movie_indices = [i[0] for i in sim_scores]
    return df['Film'].iloc[movie_indices].tolist()

# Streamlit UI
st.title("🎬 Movie Recommendation System")
selected_movie = st.selectbox("Choose a Movie", sorted(df['Film'].unique()))

if st.button("Recommend Similar Movies"):
    recommendations = get_recommendations(selected_movie)
    if recommendations:
        st.write("### Top Recommendations:")
        for movie in recommendations:
            st.write(f"- {movie}")
    else:
        st.warning("Movie not found. Try a different title.")
