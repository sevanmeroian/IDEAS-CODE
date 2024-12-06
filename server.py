import streamlit as st
import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Dataset paths
folder_path = r'C:\Users\flipp\source\repos\Final Project\Final Project'
true_file = os.path.join(folder_path, 'True.csv')
fake_file = os.path.join(folder_path, 'Fake.csv')

# Load and preprocess dataset
@st.cache_data
def load_and_prepare_data():
    true_df = pd.read_csv(true_file)
    fake_df = pd.read_csv(fake_file)

    true_df['label'] = 1  # 1 for True News
    fake_df['label'] = 0  # 0 for Fake News
    combined_df = pd.concat([true_df, fake_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    combined_df = combined_df.dropna(subset=['text'])
    return combined_df

# Train model and save it
def train_model():
    combined_df = load_and_prepare_data()
    X = combined_df['text']
    y = combined_df['label']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # Train model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_tfidf, y_train)

    # Save model, vectorizer, and test data
    with open('news_model.pkl', 'wb') as f:
        pickle.dump((model, vectorizer, X_test_tfidf, y_test), f)

    return model, vectorizer, X_test_tfidf, y_test

# Load model if already trained
def load_model():
    try:
        with open('news_model.pkl', 'rb') as f:
            return pickle.load(f)  # Returns model, vectorizer, X_test_tfidf, y_test
    except FileNotFoundError:
        return train_model()

# Streamlit app UI
st.title("News Credibility Analyzer")
st.subheader("Enter a news article to analyze its credibility")

# Input section
user_input = st.text_area("Paste your news article here:")

# Load model, vectorizer, and test data
model, vectorizer, X_test_tfidf, y_test = load_model()

if st.button("Analyze News"):
    if user_input.strip():
        # Predict credibility
        input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict_proba(input_tfidf)[0]
        credibility_score = prediction[1] * 100  # Probability of being true news

        # Display result
        st.metric(label="Credibility Score", value=f"{credibility_score:.2f}%")
        if credibility_score > 50:
            st.success("This news article is likely credible.")
        else:
            st.error("This news article is likely not credible.")
    else:
        st.warning("Please enter some text to analyze!")

# Model evaluation section
if st.checkbox("Show Model Evaluation"):
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {accuracy:.2f}")
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))
