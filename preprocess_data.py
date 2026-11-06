import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import zipfile

# Ensure you have the necessary NLTK data downloaded
nltk.download('stopwords')
nltk.download('wordnet')

# Use a context manager to open the zip file and read the data from the specific CSVs inside
with zipfile.ZipFile('archive (3).zip') as z:
    with z.open('train.csv') as f:
        # Correct the column names to match the data: 'label' is first, 'text' is second
        df_train = pd.read_csv(f, sep='\t', names=['label', 'text'])
    with z.open('test.csv') as f:
        # Do the same for the test data
        df_test = pd.read_csv(f, sep='\t', names=['label', 'text'])

# Initialize the lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # The rest of your preprocessing function remains the same
    # 1. Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # 2. Remove punctuation and special characters, and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    # 3. Tokenize
    tokens = text.split()
    # 4. Remove stopwords and perform lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # 5. Join the tokens back into a single string
    return ' '.join(tokens)

# Apply the function to your text data
df_train['clean_text'] = df_train['text'].apply(preprocess_text)
df_test['clean_text'] = df_test['text'].apply(preprocess_text)

# You can display the cleaned text to verify
print("Original Train Text:")
print(df_train['text'].iloc[0])
print("\nCleaned Train Text:")
print(df_train['clean_text'].iloc[0])

print("\nOriginal Test Text:")
print(df_test['text'].iloc[0])
print("\nCleaned Test Text:")
print(df_test['clean_text'].iloc[0])
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Assuming df_train and df_test are already loaded and pre-processed

# Initialize the TfidfVectorizer
# You can adjust parameters like max_features, min_df, etc., for optimization.
vectorizer = TfidfVectorizer(max_features=5000)

# Fit the vectorizer on the training data and transform both train and test data
X_train = vectorizer.fit_transform(df_train['clean_text'])
X_test = vectorizer.transform(df_test['clean_text'])

# The labels (target variable) are already numerical
y_train = df_train['label']
y_test = df_test['label']

print("\n--- Feature Extraction Complete ---")
print("Shape of X_train (features):", X_train.shape)
print("Shape of y_train (labels):", y_train.shape)
print("Shape of X_test (features):", X_test.shape)
print("Shape of y_test (labels):", y_test.shape)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Assuming X_train, X_test, y_train, and y_test are already created

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model on the training data
print("\n--- Training Model ---")
model.fit(X_train, y_train)
print("Model training complete.")

# Make predictions on the test data
print("\n--- Making Predictions ---")
y_pred = model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Accuracy of the Logistic Regression model: {accuracy:.4f}")
import joblib

# Assuming 'model' and 'vectorizer' have been created and trained
# Save the model and the vectorizer to disk
joblib.dump(model, 'sentiment_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

print("\n--- Model Persistence Complete ---")
print("Model and vectorizer saved to sentiment_model.joblib and tfidf_vectorizer.joblib")