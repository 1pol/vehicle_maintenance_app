# 🚗 Vehicle Maintenance Prediction App

This is a machine learning-powered web application that predicts whether a vehicle is likely to need maintenance based on user-provided data. The app is built with **Flask** and uses an **XGBoost classifier** trained on detailed vehicle service and condition data.

---

## 🔧 Features

- ✅ Real-time vehicle maintenance prediction
- 📊 Input fields include mileage, part condition, service history, fuel efficiency, and more
- 🧠 Uses a trained XGBoost model with preprocessing (ordinal + one-hot encoding)
- 🖥️ Interactive front-end using HTML templates
- ☁️ Ready for deployment on **Render**

---

## 🚀 Live Demo

🔗 [Deployed App on Render](https://vehicle-maintenance-app.onrender.com)  
_(Replace this link once deployed)_

---

## 🧰 Tech Stack

| Layer       | Technology        |
|-------------|-------------------|
| Language    | Python            |
| Web Server  | Flask + Gunicorn  |
| ML Model    | XGBoost Classifier|
| Preprocessing | scikit-learn (OrdinalEncoder, StandardScaler) |
| Deployment  | Render            |

---

## 📂 Project Structure

vehicle_maintenance_app/
├── app.py # Flask app with route logic
├── predict_utils.py # Preprocessing functions
├── train_model.py # One-time model training script
├── xgb_model.pkl # Trained ML model
├── scaler.pkl # Feature scaler
├── ordinal_encoder.pkl # Ordinal encoder
├── feature_names.pkl # Feature order used in training
├── requirements.txt # Dependencies
├── Procfile # For Render deployment
├── templates/
│ └── index.html # Frontend form
├── updated_vehicle_data.csv # Training data
├── eda_target_distribution.png # EDA chart
└── eda_correlation_heatmap.png # EDA chart
---

## 🛠️ How It Works

1. **User Input**: Vehicle and part details are entered via web form.
2. **Preprocessing**: The input is scaled, encoded, and aligned to match training features.
3. **Prediction**: The model returns a binary classification:
   - **Yes** → Vehicle likely needs maintenance
   - **No** → Vehicle is in good condition

---

## 🧪 Training Process

- Training data is cleaned, date features are engineered, and categorical features are encoded.
- An **XGBoost Classifier** is trained and evaluated.
- Artifacts (model, scaler, encoder, feature names) are saved using `pickle`.

---

## 📦 Installation (Local)

```bash
git clone https://github.com/your-username/vehicle_maintenance_app.git
cd vehicle_maintenance_app
pip install -r requirements.txt
python app.py



---
👨‍💻 Author
Harshit Polmersetty
Made with 💻, ☕, and ML.
