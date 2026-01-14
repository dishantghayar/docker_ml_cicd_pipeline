import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Load dataset
data = pd.read_csv("spam.csv", encoding="latin-1")[["v1", "v2"]]
data.columns = ["label", "message"]

# Create vectorizer OBJECT (not class)
vectorizer = CountVectorizer()

# Transform text
X = vectorizer.fit_transform(data["message"])
y = data["label"]

# Train model
model = MultinomialNB()
model.fit(X, y)

# Save artifacts
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(vectorizer, open("models/vectorizer.pkl", "wb"))
print("Model trained and saved") 