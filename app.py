from flask import Flask, request, jsonify, render_template
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load the trained model and vectorizer
model = joblib.load('sentiment_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

# Initialize Flask app
app = Flask(__name__)

# Re-define the preprocessing function from your training script
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

@app.route('/')
def home():
    # This route will render the main HTML page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the movie review from the web form
    review = request.json['review']
    
    # Preprocess the new review
    cleaned_review = preprocess_text(review)
    
    # Transform the cleaned review into a numerical vector using the saved vectorizer
    vectorized_review = vectorizer.transform([cleaned_review])
    
    # Make a prediction using the saved model
    prediction = model.predict(vectorized_review)
    
    # Get the predicted sentiment as a string
    sentiment = 'Positive' if prediction[0] == 1 else 'Negative'
    
    # Return the prediction as a JSON response
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    # Run the web application
    app.run(debug=True)