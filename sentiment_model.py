import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import pickle

# Sample data (replace this with a full dataset like IMDB or Amazon reviews)
data = {
    "review": [
        "I love this product! It's amazing!",
        "Worst experience ever. Not worth the money.",
        "It's okay, not great but not bad.",
        "Absolutely fantastic, will buy again.",
        "Terrible product, broke in a week.",
        "Not bad, but I expected more.",
        "Highly recommended, very satisfied.",
        "Quality is poor and service is bad.",
        "Decent value for the price.",
        "This product exceeded my expectations!"
    ],
    "sentiment": [
        "positive", "negative", "neutral", "positive", "negative",
        "neutral", "positive", "negative", "neutral", "positive"
    ]
}

df = pd.DataFrame(data)

# Text preprocessing + TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['review'])
y = df['sentiment']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model and vectorizer
pickle.dump(model, open("sentiment_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))
