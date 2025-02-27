from flask import Flask, request, jsonify, send_file
import joblib
import pandas as pd
import io

app = Flask(__name__)

# Load trained model and scaler
try:
    modelo = joblib.load("final_model.pkl")
    scaler = joblib.load("scaler.pkl")
    print("✅ Model and scaler loaded successfully")
except Exception as e:
    print(f"❌ Error loading model or scaler: {e}")

# Expected categorical and numerical columns
categorical_columns = ['Income_type', 'Education_type', 'Family_status', 'Housing_type', 'Occupation_type']
numerical_features = ['Account_length', 'Total_income', 'Age', 'Years_employed']
extra_features = ["Gender", "Own_car", "Own_property", "Work_phone", "Phone", "Email", "Unemployed", "Num_children", "Num_family"]

@app.route("/", methods=["GET"])
def home():
    return "✅ Prediction API is running on Render!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        
        # Apply One-Hot Encoding
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=False)
        
        # Ensure all expected columns are present
        expected_columns = list(scaler.feature_names_in_)
        missing_cols = set(expected_columns) - set(df.columns)
        for col in missing_cols:
            df[col] = 0  # Fill missing columns with 0
        
        # Ensure correct column order
        df = df[expected_columns]

        # Save logs to a file in Render
        with open("logs.txt", "w") as log_file:
            log_file.write(f"🔹 Columns received in API: {df.columns.tolist()}\n")
            log_file.write(f"🔹 Columns expected by model: {expected_columns}\n")
            log_file.write(f"🔹 Total columns received: {len(df.columns)}\n")
            log_file.write(f"🔹 Total columns expected: {len(expected_columns)}\n")
        
        # Print logs in Render logs output
        with open("logs.txt", "r") as log_file:
            print("\n📌 DEBUG LOGS FROM logs.txt:\n" + log_file.read())

        # Apply MinMax Scaling
        df[numerical_features] = scaler.transform(df[numerical_features])
        
        # Make the prediction
        prediction = modelo.predict(df)[0]
        return jsonify({"prediction": int(prediction)})
    
    except Exception as e:
        print(f"❌ Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
