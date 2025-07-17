import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier

# 1. Load Dataset
df = pd.read_csv("updated_vehicle_data.csv")

# 2. Handle Missing Values
df = df.dropna(subset=["Need_Maintenance"])  # Target must not be missing

# 3. Convert Date Columns
df["Last_Service_Date"] = pd.to_datetime(df["Last_Service_Date"])
df["Warranty_Expiry_Date"] = pd.to_datetime(df["Warranty_Expiry_Date"])

today = pd.Timestamp.today()

df["Days_Since_Last_Service"] = (today - df["Last_Service_Date"]).dt.days
df["Days_Until_Warranty_Expires"] = (df["Warranty_Expiry_Date"] - today).dt.days

df.drop(["Last_Service_Date", "Warranty_Expiry_Date"], axis=1, inplace=True)

# 4. Ordinal Encoding
ordinal_cols = ["Maintenance_History", "Tire_Condition", "Brake_Condition", "Battery_Status"]
ordinal_mapping = [
    ["Poor", "Average", "Good"],      # Maintenance_History
    ["Worn Out", "Good", "New"],      # Tire_Condition
    ["Worn Out", "Good", "New"],      # Brake_Condition
    ["Weak", "Good", "New"]           # Battery_Status
]
ordinal_encoder = OrdinalEncoder(categories=ordinal_mapping)
df[ordinal_cols] = ordinal_encoder.fit_transform(df[ordinal_cols])

# 5. One-Hot Encoding
nominal_cols = ["Vehicle_Model", "Fuel_Type", "Transmission_Type", "Owner_Type"]
df = pd.get_dummies(df, columns=nominal_cols)

# 6. Split Features and Target
X = df.drop("Need_Maintenance", axis=1)
y = df["Need_Maintenance"]

# Save feature names
with open("feature_names.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

# 7. EDA (Optional but Useful)
# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_correlation_heatmap.png")
plt.show()
plt.close()

# Target balance
plt.figure(figsize=(4, 3))
sns.countplot(x="Need_Maintenance", data=df)
plt.title("Target Distribution")
plt.tight_layout()
plt.savefig("eda_target_distribution.png")
plt.show()
plt.close()

# 8. Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 9. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 10. Train XGBoost Classifier
model = XGBClassifier(eval_metric="logloss")
model.fit(X_train, y_train)

# 11. Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# 12. Save Model Artifacts
pickle.dump(model, open("xgb_model.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))
pickle.dump(ordinal_encoder, open("ordinal_encoder.pkl", "wb"))