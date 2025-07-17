from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from predict_utils import preprocess_input

app = Flask(__name__)

# Load model and tools
model = pickle.load(open("xgb_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
ordinal_encoder = pickle.load(open("ordinal_encoder.pkl", "rb"))
feature_names = pickle.load(open("feature_names.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        form_data = request.form.to_dict()
        try:
            processed_input = preprocess_input(form_data, ordinal_encoder, scaler, feature_names)
            prediction = model.predict(processed_input)[0]
            result = "Yes" if prediction == 1 else "No"
        except Exception as e:
            result = f"Error: {str(e)}"
        return render_template("index.html", result=result)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
