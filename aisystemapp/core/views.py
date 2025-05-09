# core/views.py
from django.shortcuts import render
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os

# Load models
sentiment_model = tf.keras.models.load_model('sentiment_model.h5')
title_model = tf.keras.models.load_model('title_model.h5')

# Dummy tokenizer setup (replace with your own tokenizer loading)
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer_input = Tokenizer(oov_token="<OOV>")
tokenizer_output = Tokenizer(oov_token="<OOV>")
# load your tokenizer configurations or fit it

MAX_INPUT_LEN = 300
MAX_TARGET_LEN = 20

def predict_view(request):
    result = {}
    if request.method == 'POST':
        text = request.POST.get('review')

        # Preprocess text
        seq = tokenizer_input.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=MAX_INPUT_LEN, padding='post')

        # Sentiment prediction
        sentiment = sentiment_model.predict(padded)[0]
        sentiment_label = 'Positive' if sentiment > 0.5 else 'Negative'

        # Title generation
        decoder_input = np.zeros((1, MAX_TARGET_LEN))  # Replace with your generation logic
        title = "Generated Title"  # Placeholder

        result = {
            'text': text,
            'sentiment': sentiment_label,
            'title': title
        }

    return render(request, 'core/index.html', result)
