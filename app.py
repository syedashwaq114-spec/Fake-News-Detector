import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import streamlit as st

# Load datasets
fake = pd.read_csv("Fake.csv")
true = pd.read_csv("True.csv")

# Labels
fake["label"] = 0
true["label"] = 1

# Combine datasets
data = pd.concat([fake, true])
data["content"] = data["title"] + " " + data["text"]
data["content"] = data["content"].str.lower() 

# Input and output
x = data["content"]
y = data["label"]

# Convert text to numbers
vectorizer = TfidfVectorizer(stop_words='english',max_df=0.7)
x = vectorizer.fit_transform(x)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Train model
model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(x_train, y_train)

# Accuracy
pred = model.predict(x_test)
accuracy = accuracy_score(y_test, pred)

# Streamlit UI
st.title("Fake News Detector")

st.write("Model Accuracy:", accuracy)

news = st.text_area("Enter News Article")

if st.button("Predict"):
    news_vector = vectorizer.transform([news])
    result = model.predict(news_vector)

    if result[0] == 1:
        st.success("This is REAL News")
    else:
        st.error("This is FAKE News")

