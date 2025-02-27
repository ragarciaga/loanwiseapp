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
all_features = ["Gender", "Own_car", "Own_property", "Work_phone", "Phone", "Email", "Unemployed", "Num_children", "Num_family"] + numerical_features + categorical_columns

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
        missing_cols = set(scaler.feature_names_in_) - set(df.columns)
        for col in missing_cols:
            df[col] = 0  # Fill missing columns with 0

        df = df[scaler.feature_names_in_]  # Arrange columns in the same order

        # Apply MinMax Scaling
        df[numerical_features] = scaler.transform(df[numerical_features])
        
        # Make the prediction
        prediction = modelo.predict(df)[0]
        return jsonify({"prediction": int(prediction)})
    
    except Exception as e:
        print(f"❌ Error in /predict: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/predict_csv", methods=["POST"])
def predict_csv():
    try:
        if "file" not in request.files:
            return jsonify({"error": "No file provided with key 'file'"}), 400

        file = request.files["file"]
        df = pd.read_csv(file)
        
        # Apply One-Hot Encoding
        df = pd.get_dummies(df, columns=categorical_columns, drop_first=False)
        
        # Ensure all expected columns are present
        missing_cols = set(scaler.feature_names_in_) - set(df.columns)
        for col in missing_cols:
            df[col] = 0  # Fill missing columns with 0

        df = df[scaler.feature_names_in_]  # Arrange columns in the same order

        # Apply MinMax Scaling
        df[numerical_features] = scaler.transform(df[numerical_features])

        # Make predictions
        df["Target"] = modelo.predict(df)

        # Convert to CSV format
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)

        return send_file(
            io.BytesIO(output.getvalue().encode()),
            mimetype="text/csv",
            as_attachment=True,
            attachment_filename="predictions.csv"
        )
    
    except Exception as e:
        print(f"❌ Error in /predict_csv: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
