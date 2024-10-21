#import library
import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import time
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#Load Model
with open('text_segmentation_model.pkl', 'rb') as file:
    model = pickle.load(file)

with open('tfidf_vectorizer.pkl', 'rb') as file:
    tfidf = pickle.load(file)

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

st.sidebar.title("Text Sentiment Analysis")
st.sidebar.markdown("<p style='font-size: 18px;'>Use this app to classify text as positive, negative or neutral based on model predictions.</p>", unsafe_allow_html=True)

st.sidebar.markdown("### How to Use This App")
st.sidebar.write("""
1. Enter a text in the main area to get sentiment analysis.
2. Use the **Predict** button to get the sentiment result.
3. The **Clear** button will reset your input.
4. A **word cloud** will display the prominent words in your input.
""")

if "user_input" in st.session_state and st.session_state["user_input"]:
    st.sidebar.markdown("### Word Cloud of the Input Text")
    wordcloud = WordCloud(background_color='white').generate(st.session_state["user_input"])
    fig, ax = plt.subplots()
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.sidebar.pyplot(fig)

st.markdown("<h1 style='text-align: center;'>Text Sentiment Analysis</h1>", unsafe_allow_html=True)
st.markdown("<h5 style='text-align: center; margin-bottom:2%;'>Enter a text below, and the model will predict if it's negative, positive or neutral.</h5>", unsafe_allow_html=True)

if "user_input" not in st.session_state:
    st.session_state["user_input"] = ""

def clear_form():
    st.session_state["user_input"] = ""

if st.button("Clear"):
    clear_form()

with st.form("Text Form"):
    user_input = st.text_area("",value=st.session_state["user_input"],key="user_input", height=150, placeholder="Enter your text here...", max_chars=1000)
    submit = st.form_submit_button(label="Predict", type="primary")

if submit:
    if user_input == "":
        st.warning("⚠️ Please fill in the text area before submitting.")
    
    else:
        with st.spinner("Predicting..."):
            time.sleep(1)
            data = [user_input]
            data_trans = tfidf.transform(data)
            prediction = model.predict(data_trans)
            result = prediction[0]
        if result == 1:
            st.success("✅ The text expresses a favourable sentiment")
        elif result == 2:
            st.warning("✅ The text expresses a Neutral sentiment")
        else:
            st.error("⚠️ The text expresses an unfavourable sentiment.")

st.markdown("<hr style='border: 1px solid #e0e0e0; margin-top:5%;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: grey; font-size: 12px;'>Developed by Sokputthi Phonpo | © 2024</p>", unsafe_allow_html=True)


