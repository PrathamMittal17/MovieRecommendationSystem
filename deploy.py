import pickle
import tmdbsimple as tmdb
import pandas as pd
import streamlit as st
import spacy

tmdb.API_KEY = '263c6358c90d7f6db1e6d19845df32c8'
sentiment_analysis = pickle.load(open('sentiment_analysis_model.pkl', 'rb'))
movies_df = pd.read_csv("data/movies.csv")
similarity = pickle.load(open('similarity.pkl', 'rb'))
nlp = spacy.load("en_core_web_sm")

indices = pd.Series(movies_df.index, index=movies_df.title)


def recommend_movie(title, how_many_movies=5):
    recommended_movies = []
    index_title = indices[title]
    sim = pd.Series(similarity[index_title]).sort_values(ascending=False)[1:how_many_movies + 1]
    for i in sim.index:
        recommended_movies.append(indices[indices == i].index[0])
    return recommended_movies


selected_movie = st.selectbox("Choose a movie to get recommendations:", movies_df.title)
button_clicked = st.button("Recommend")

if button_clicked:
    rec_movies = recommend_movie(selected_movie)
    movies_data = {}
    data = []
    reviews = []
    for movie in rec_movies:
        data = []
        reviews = []
        id = movies_df[movies_df.title == movie]['id'].to_string(index=False)
        movie_req = tmdb.Movies(id)
        response = movie_req.info()
        poster = f"https://image.tmdb.org/t/p/original/{response['poster_path']}"
        overview = response['overview']
        response = movie_req.reviews()
        review = response['results']

        for i in review:
            reviews.append(i['content'].replace("\n", '').replace("\r", '').replace('\'', ""))

        data.append({"Poster": poster, "Overview": overview, "Reviews": reviews})
        movies_data[movie] = data

    for movie in rec_movies:
        title = movie
        st.header(title)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(movies_data[title][0]['Poster'], width=200)
        with col2:
            st.write(movies_data[title][0]['Overview'])
        if not movies_data[title][0]['Reviews']:
            st.write("No Reviews Found ")
        with st.expander("Reviews: "):
            for i in movies_data[title][0]['Reviews']:
                " ".join(
                    token.lemma_ for token in nlp(str(i))
                    if not token.is_punct and token.lemma_.lower()
                )
                sentiment = sentiment_analysis.predict([i])
                if sentiment == 1:
                    st.caption("Positive Review")
                    i = f'<span style="color:#45E00B;">{i}</span>'
                    st.markdown("• " + i, unsafe_allow_html=True)
                else:
                    st.caption("Negative Review")
                    i = f'<span style="color:Red;">{i}</span>'
                    st.markdown("• " + i, unsafe_allow_html=True)
