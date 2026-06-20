from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        news_text = request.form["news"]

        transformed_text = vectorizer.transform([news_text])
        result = model.predict(transformed_text)[0]

        prediction = "Real News ✅" if result == 1 else "Fake News ❌"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)