import numpy as np
import pandas as pd
from datetime import datetime

def preprocess_input(data, ordinal_encoder, scaler, feature_names):
    # Parse numeric fields
    numeric_fields = [
        "Mileage", "Reported_Issues", "Vehicle_Age", "Engine_Size",
        "Odometer_Reading", "Insurance_Premium", "Service_History",
        "Accident_History", "Fuel_Efficiency"
    ]
    for field in numeric_fields:
        data[field] = float(data[field])

    # Compute time-based features
    today = pd.Timestamp.today()
    data["Days_Since_Last_Service"] = (today - pd.to_datetime(data["Last_Service_Date"])).days
    data["Days_Until_Warranty_Expires"] = (pd.to_datetime(data["Warranty_Expiry_Date"]) - today).days

    # Ordinal encode
    ordinal_cols = ["Maintenance_History", "Tire_Condition", "Brake_Condition", "Battery_Status"]
    ordinal_df = pd.DataFrame([[data[col] for col in ordinal_cols]], columns=ordinal_cols)
    ordinal_encoded = ordinal_encoder.transform(ordinal_df)

    # One-hot encode
    one_hot_cols = ["Vehicle_Model", "Fuel_Type", "Transmission_Type", "Owner_Type"]
    one_hot_input = {col: data[col] for col in one_hot_cols}
    one_hot_df = pd.DataFrame([one_hot_input])
    one_hot_encoded = pd.get_dummies(one_hot_df)

    # Create base input vector
    input_dict = {col: [data[col]] for col in numeric_fields}
    input_dict["Days_Since_Last_Service"] = [data["Days_Since_Last_Service"]]
    input_dict["Days_Until_Warranty_Expires"] = [data["Days_Until_Warranty_Expires"]]

    base_df = pd.DataFrame(input_dict)
    for idx, col in enumerate(ordinal_cols):
        base_df[col] = ordinal_encoded[:, idx]

    # Concatenate one-hot columns
    full_df = pd.concat([base_df, one_hot_encoded], axis=1)

    # Add any missing columns (must match training columns)
    for col in feature_names:
        if col not in full_df.columns:
            full_df[col] = 0

    # Reorder columns
    full_df = full_df[feature_names]

    # Scale
    scaled_input = scaler.transform(full_df)
    return scaled_input
