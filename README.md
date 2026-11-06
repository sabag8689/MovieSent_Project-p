# MovieSent – Movie Review Sentiment Analysis

MovieSent is a web application that analyzes a written movie review and predicts whether its sentiment is **Positive** or **Negative**.

## Features

- Enter a movie review through a simple web interface.
- Preprocess text using NLTK.
- Convert text into numerical features using TF-IDF.
- Predict sentiment using a trained Logistic Regression model.
- Display the prediction in the browser using a Flask backend.

## Technologies

- Python
- Flask
- Scikit-learn
- NLTK
- TF-IDF
- Logistic Regression
- HTML, CSS and JavaScript

## Project structure

```text
MovieSent/
├── app.py
├── preprocess_data.py
├── sentiment_model.joblib
├── tfidf_vectorizer.joblib
├── archive (3).zip
├── requirements.txt
├── README.md
├── .gitignore
└── templates/
    └── index.html
```

## Setup and run

1. Clone this repository:

   ```bash
   git clone YOUR_GITHUB_REPOSITORY_URL
   cd MovieSent
   ```

2. Create and activate a virtual environment (recommended):

   **Windows PowerShell**
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

   **macOS / Linux**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Download the NLTK resources required by the application:

   ```bash
   python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
   ```

5. Start the Flask application:

   ```bash
   python app.py
   ```

6. Open the local address shown in the terminal (normally `http://127.0.0.1:5000/`).

## Model files

The repository includes the trained Logistic Regression model and its fitted TF-IDF vectorizer. Both are required by `app.py`.

## Training

The `preprocess_data.py` script contains the data loading, text preprocessing, TF-IDF feature extraction, Logistic Regression training, evaluation, and model-saving workflow. It expects the original dataset archive (`archive (3).zip`) to be in the project root.

## Notes

- This project is intended for local demonstration and learning.
- The Flask development server uses debug mode in the current application. Do not use the development server or debug mode for public production deployment.
- Dataset files are included as an archive to avoid keeping duplicate extracted CSV files in the repository.
