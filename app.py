from flask import Flask, request, jsonify
import pickle


app = Flask(__name__)
model = pickle.load(open("models/model.pkl", "rb"))

vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))


@app.route("/")
def home():
    return "ML Spam Classifier API is running"

@app.route("/predict", methods=["POST"])
def predict():
    message = request.json["message"]
    vector = vectorizer.transform([message])
    prediction = model.predict(vector)[0]
    return jsonify({"prediction":prediction})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000) 