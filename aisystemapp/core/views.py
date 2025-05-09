# core/views.py
from django.shortcuts import render
from django.conf import settings
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import os
import pickle


# Construct the path
MODEL_DIR = os.path.join(settings.BASE_DIR, 'tf_models')


# Load tokenizers
with open(os.path.join(MODEL_DIR, 'tokenizer_input.pkl'), 'rb') as f:
    tokenizer_input = pickle.load(f)

with open(os.path.join(MODEL_DIR, 'tokenizer_output.pkl'), 'rb') as f:
    tokenizer_output = pickle.load(f)

# Load sentiment analysis model
with open(os.path.join(MODEL_DIR, 'sentiment_prediction_model.pkl'), 'rb') as f:
    sentiment_model = pickle.load(f)

# Load sentiment analysis vectorizer
with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'rb') as f:
    sentiment_vectorizer = pickle.load(f)


# Load the title generation model
title_model = tf.keras.models.load_model(os.path.join(MODEL_DIR, 'title_prediction_model.h5'))

# Title generation function
MAX_INPUT_LEN = 300
MAX_TARGET_LEN = 20

MAX_INPUT_LEN = 300
MAX_TARGET_LEN = 20

def generate_title(text):
    sequence = tokenizer_input.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_INPUT_LEN, padding='post')
    output = np.zeros((1, MAX_TARGET_LEN))
    output[0, 0] = tokenizer_output.word_index['<start>'] if '<start>' in tokenizer_output.word_index else 1

    for i in range(1, MAX_TARGET_LEN):
        prediction = title_model.predict([padded, output], verbose=0)
        next_word_id = np.argmax(prediction[0, i - 1])
        output[0, i] = next_word_id
        if next_word_id == tokenizer_output.word_index.get('<end>', 2):
            break

    predicted_sequence = [int(x) for x in output[0]]
    predicted_words = [tokenizer_output.index_word.get(i, '') for i in predicted_sequence if i != 0]
    return ' '.join(predicted_words).replace('<start>', '').replace('<end>', '').strip()

def index(request):
    sentiment = None
    title = None
    user_input = None

    if request.method == "POST":
        user_input = request.POST.get("user_input")
        if user_input:
            # Sentiment prediction
            transformed = sentiment_vectorizer.transform([user_input])
            sentiment = sentiment_model.predict(transformed)[0]

            # Title prediction
            title = generate_title(user_input)
            title = title.replace('<OOV>', '').strip().title()

    return render(request, "index.html", {
        "sentiment": sentiment,
        "title": title,
        "user_input": user_input
    })