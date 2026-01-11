# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb.h5')

# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


import streamlit as st
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page title
st.set_page_config(page_title="IMDB Sentiment Analysis", page_icon="🎬", layout="centered")

st.title('🎬 IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review below, and the model will classify it as **Positive** or **Negative**.')

# User input
user_input = st.text_area('✍️ Enter your movie review here:', height=150)

# Button
if st.button('✅ Classify Review'):
    if user_input.strip() == "":
        st.warning("Please enter a review before clicking Classify!")
    else:
        # Preprocess input
        preprocessed_input = preprocess_text(user_input)  # ensure this returns padded sequence

        # Make prediction
        prediction = model.predict(preprocessed_input, verbose=0)
        sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
        score = prediction[0][0]

        # Display results in styled format
        st.markdown(f"### Sentiment: {'✅ Positive' if sentiment=='Positive' else '❌ Negative'}")
        st.progress(float(score))  # shows a progress bar for the prediction score
        st.write(f"**Prediction Score:** {score:.3f}")

else:
    st.info('Please enter a movie review and click the **Classify Review** button.')


