from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and feature names
model = joblib.load("model.pkl")
columns = joblib.load("columns.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {}

    # Numeric features
    numeric_features = [
        "age",
        "flight distance",
        "departure delay",
        "arrival delay",
        "departure and arrival time convenience",
        "ease of online booking",
        "check-in service",
        "online boarding",
        "gate location",
        "on-board service",
        "seat comfort",
        "leg room service",
        "cleanliness",
        "food and drink",
        "in-flight service",
        "in-flight wifi service",
        "in-flight entertainment",
        "baggage handling"
    ]

    for feature in numeric_features:
        data[feature] = float(request.form[feature])
    if data["age"] < 1 or data["age"] > 105:
        return render_template(
            "index.html",
            prediction="Age must be between 1 and 105 years.",
            confidence=""
        )

    if data["flight distance"] < 0:
        return render_template(
            "index.html",
            prediction="Flight distance cannot be negative.",
            confidence=""
        )

    if data["departure delay"] < 0:
        return render_template(
            "index.html",
            prediction="Departure delay cannot be negative.",
            confidence=""
        )

    if data["arrival delay"] < 0:
        return render_template(
            "index.html",
            prediction="Arrival delay cannot be negative.",
            confidence=""
        )
    # Gender
    data["gender_Male"] = 1 if request.form["gender"] == "Male" else 0

    # Customer Type
    data["customer type_Returning"] = 1 if request.form["customer_type"] == "Returning" else 0

    # Travel Type
    data["type of travel_Personal"] = 1 if request.form["travel_type"] == "Personal" else 0

    # Travel Class
    travel_class = request.form["travel_class"]

    data["class_Economy"] = 1 if travel_class == "Economy" else 0
    data["class_Economy Plus"] = 1 if travel_class == "Economy Plus" else 0

    # Create DataFrame
    df = pd.DataFrame([data])

    # Match training columns
    df = df.reindex(columns=columns, fill_value=0)

    # Prediction
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0]
    confidence = round(max(probability) * 100,2)

    if prediction == 1:
        result = "Satisfied"
    else:
        result = "Neutral or Dissatisfied"

    return render_template("index.html", prediction=result,confidence=confidence)


if __name__ == "__main__":
    app.run(debug=True)