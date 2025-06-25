import streamlit as st
import pickle
import numpy as np

# Load model and vectorizer
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# Map for display
sentiment_map = {
    'positive': 'Positive 😊',
    'neutral': 'Neutral 😐',
    'negative': 'Negative 😞'
}

# Title
st.title("🧠 Product Review Sentiment Analyzer")

# User input
user_review = st.text_area("Enter Product Review:")

if st.button("Analyze Sentiment"):
    if user_review.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Transform input
        input_vector = vectorizer.transform([user_review])
        prediction = model.predict(input_vector)[0]
        proba = model.predict_proba(input_vector).max() * 100
        
        # Output
        st.subheader("Predicted Sentiment:")
        st.success(f"{sentiment_map[prediction]} (Confidence: {proba:.2f}%)")
