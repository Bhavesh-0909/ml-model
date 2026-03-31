import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib
from datasets import load_dataset

# Load the dataset from Hugging Face
print("Loading dataset...")
dataset = load_dataset("locuoco/the-biggest-spam-ham-phish-email-dataset-300000")

# 1. Convert the 'train' split into a pandas DataFrame
print("Converting to pandas DataFrame...")
df = dataset["train"].to_pandas()

# Data Cleaning: Remove missing values and enforce string types
print("Cleaning data...")
df = df.dropna(subset=['text', 'label'])
df['text'] = df['text'].astype(str)

print(f"Dataset ready. Total rows: {len(df)}")

# 2. Split the data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# 3. Create a Pipeline (Vectorization + Classification)
print("Building pipeline...")
model_pipeline = make_pipeline(
    TfidfVectorizer(stop_words='english', max_features=5000), 
    MultinomialNB()
)

# 4. Train the model
print("Training model (this might take a moment depending on dataset size)...")
model_pipeline.fit(X_train, y_train)

print("Scoring model...")
accuracy = model_pipeline.score(X_test, y_test)
print(f"Model trained with accuracy: {accuracy * 100:.2f}%")

# 5. Save the trained pipeline to disk
joblib.dump(model_pipeline, 'spam_classifier.pkl')
print("Model saved as spam_classifier.pkl")