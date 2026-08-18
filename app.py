import streamlit as st
import pandas as pd
import re
import joblib


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="IMDb Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)


# --------------------------------------------------
# Load Saved Model and TF-IDF Vectorizer
# --------------------------------------------------

@st.cache_resource
def load_models():
    model = joblib.load("sentiment_model.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")

    return model, tfidf


model, tfidf = load_models()


# --------------------------------------------------
# Text Cleaning
# --------------------------------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(
        r"[^a-zA-Z0-9\s']",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("🎬 IMDb Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review below and the machine learning "
    "model will predict whether the sentiment is positive or negative."
)


review = st.text_area(
    "Enter your movie review:",
    height=150,
    placeholder="Example: This movie was absolutely amazing!"
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a movie review.")

    else:

        # Clean review
        cleaned_review = clean_text(review)

        # Convert text to TF-IDF
        review_tfidf = tfidf.transform([cleaned_review])

        # Prediction
        prediction = model.predict(review_tfidf)[0]

        # Probability
        probabilities = model.predict_proba(review_tfidf)[0]

        # Map probabilities to class names
        class_probabilities = dict(
            zip(model.classes_, probabilities)
        )

        positive_probability = class_probabilities["positive"]
        negative_probability = class_probabilities["negative"]


        # --------------------------------------------------
        # Display Result
        # --------------------------------------------------

        st.subheader("Prediction Result")


        if prediction == "positive":

            st.success("😊 Positive Sentiment")

        else:

            st.error("😞 Negative Sentiment")


        # Display probabilities

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Positive Probability",
                f"{positive_probability:.2%}"
            )

        with col2:

            st.metric(
                "Negative Probability",
                f"{negative_probability:.2%}"
            )