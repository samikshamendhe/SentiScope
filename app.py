from flask import Flask, render_template, request
import pickle
import re

app = Flask(__name__)

# Load Model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))


# -----------------------------
# Clean Review
# -----------------------------
def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z ]", "", text)

    return text


# -----------------------------
# Predict Review
# -----------------------------
def predict_review(review):

    review = clean_text(review)

    review_vector = vectorizer.transform([review])

    probability = model.predict_proba(review_vector)[0]

    classes = model.classes_

    if max(probability) < 0.60:
        return "Neutral"

    return classes[probability.argmax()]


# -----------------------------
# Predict Rating
# -----------------------------
def predict_star(stars):

    if stars <= 2:
        return "Negative"

    elif stars == 3:
        return "Neutral"

    else:
        return "Positive"


# -----------------------------
# Home Page
# -----------------------------
@app.route("/")
def home():

    return render_template("index.html")


# -----------------------------
# Prediction
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():

    review = request.form["review"]

    stars = int(request.form["stars"])

    review_prediction = predict_review(review)

    rating_prediction = predict_star(stars)

    final_result = review_prediction

    if final_result == "Positive":

        emoji = "😊"

        color = "#22c55e"

        message = "Customer is Happy!"

    elif final_result == "Negative":

        emoji = "😞"

        color = "#ef4444"

        message = "Customer is Unhappy!"

    else:

        emoji = "😐"

        color = "#FFD700"

        message = "Customer has Mixed Feelings."

    return render_template(

        "index.html",

        prediction=final_result,

        emoji=emoji,

        color=color,

        message=message,

        review=review,

        stars=stars

    )


# -----------------------------
# Run Flask
# -----------------------------
if __name__ == "__main__":

    app.run(debug=True)