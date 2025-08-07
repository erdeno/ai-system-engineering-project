# core/views.py
from django.shortcuts import render
from django.conf import settings
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

import numpy as np
import os
import pickle


# Construct the path
MODEL_DIR = os.path.join(settings.BASE_DIR, 'tf_models')

# Load sentiment analysis model
with open(os.path.join(MODEL_DIR, 'sentiment_prediction_model.pkl'), 'rb') as f:
    sentiment_model = pickle.load(f)

# Load sentiment analysis vectorizer
with open(os.path.join(MODEL_DIR, 'vectorizer.pkl'), 'rb') as f:
    sentiment_vectorizer = pickle.load(f)

# Load T5 Model. T5 stands for Text-To-Text Transfer Transformer.
MODEL_DIR = "aisystemapp/tf_models/t5_model"
TOKENIZER_DIR = "aisystemapp/tf_models/t5_tokenizer"

def get_t5_model_and_tokenizer():
    # Check if model and tokenizer are already saved
    if not (os.path.exists(MODEL_DIR) and os.path.exists(TOKENIZER_DIR)):
        print("Downloading and saving T5-small model and tokenizer...")
        tokenizer = T5Tokenizer.from_pretrained("t5-small")
        model = T5ForConditionalGeneration.from_pretrained("t5-small")

        # Create directory structure
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(TOKENIZER_DIR, exist_ok=True)

        # Save locally
        tokenizer.save_pretrained(TOKENIZER_DIR)
        model.save_pretrained(MODEL_DIR)
    else:
        print("Loading saved T5-small model and tokenizer...")
        tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_DIR)
        model = T5ForConditionalGeneration.from_pretrained(MODEL_DIR)

    return model, tokenizer


t5_model, t5_tokenizer = get_t5_model_and_tokenizer()

def generate_title_t5(text):
    input_text = f"summarize: {text}"
    inputs = t5_tokenizer.encode(input_text, return_tensors='pt', max_length=len(text), truncation=True)
    
    # Generate the title
    outputs = t5_model.generate(inputs, max_length=20, num_beams=8, do_sample=True, early_stopping=True)
    title = t5_tokenizer.decode(outputs[0], skip_special_tokens=True)

    title = title.split('.')[0].capitalize()
    
    return title

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
            title = generate_title_t5(user_input)

    return render(request, "index.html", {
        "sentiment": sentiment,
        "title": title,
        "user_input": user_input
    })