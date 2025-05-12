# 📝 Project: Sentiment & Title Generator Web App

This is a Django-based web application that takes a user's text input (e.g., a product review), performs **sentiment analysis** (positive or negative), and generates a **suitable title** using pre-trained machine learning models.

---

## 📦 Features

- Predicts **sentiment** (Positive or Negative) using a `.pkl` model.
- Generates a relevant **title** using the t5-small model from HuggingFace Transformers
- Stylish UI with emojis reflecting sentiment.
- Clean and modern 2-panel layout for results.



You can see our [MLFlow](https://dagshub.com/ramin.kazemi91/AIproject.mlflow/#/experiments/0/runs/f0e4f255749f4d138e33e4afa1bc8ac8) metrics in your browser.

---

## ⚙️ Setup Instructions

### 1. ✅ Clone the Repository

```bash
git clone https://github.com/erdeno/ai-system-engineering-project.git
cd aisystemapp
```

### 2. 🐍 Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate     # On Linux/macOS
venv\Scripts\activate        # On Windows
```

### 3. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

> 💡 If you encounter errors, make sure `pip` and `setuptools` are up to date:
> `pip install --upgrade pip setuptools`

---

## 🧠 Pre-trained Models

Ensure the following files are present in the `tf_models/` folder (in the root directory):

- `sentiment_model.pkl` – Sentiment classification model
- `vectorizer.pkl` – TF-IDF or CountVectorizer for sentiment model
- `title_model.h5` – Title generation model (TensorFlow)
- `tokenizer_input.pkl` – Tokenizer used to encode content text
- `tokenizer_output.pkl` – Tokenizer used to decode predicted title

If not already present, copy your model files there.

---

## 🚀 Run the App

```bash
python manage.py migrate
python manage.py runserver
```

Now visit [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.

---

## 🧪 Running Tests

To ensure everything works properly:

```bash
python manage.py test
```

---

## 📁 Project Structure

```
.
├── core/
│   ├── templates/core/
│   │   └── index.html
│   ├── views.py
│   ├── urls.py
│   └── tests.py
├── tf_models/
│   ├── t5_model/               # Saved T5 model (auto-downloaded if not present)
│   └── t5_tokenizer/           # Saved T5 tokenizer
│   ├── sentiment_prediction_model.pkl
│   └── vectorizer.pkl
├── manage.py
├── requirements.txt
└── README.md
```

---

## ❓ FAQ

**Q: I get "TemplateDoesNotExist" error. What should I do?**  
Make sure your `index.html` file is located in `core/templates/core/`.

**Q: Can I run this on Windows/Linux/macOS?**  
Yes! This app is OS-independent and works in any Python environment with the right packages installed.

**Q: Can I deploy this on Heroku or Render?**  
Absolutely — you'll need to add a `Procfile`, `runtime.txt`, and possibly switch to a production-ready database.

