from fastapi import FastAPI, File, UploadFile
import pandas as pd
import pickle
import uvicorn
from pydantic import BaseModel
import io
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Load the trained model
with open("model_experiment_9.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize the API
app = FastAPI()

# Define input schema for JSON
class InputData(BaseModel):
    Gender: int
    Own_car: int
    Own_property: int
    Work_phone: int
    Phone: int
    Email: int
    Unemployed: int
    Num_children: int
    Num_family: int
    Account_length: int
    Total_income: float
    Age: float
    Years_employed: float
    Income_type: str
    Education_type: str
    Family_status: str
    Housing_type: str
    Occupation_type: str

# Preprocessing function
def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess input data before making predictions:
    - Apply One-Hot Encoding for categorical variables
    - Fill missing values with median
    - Normalize numerical variables using MinMaxScaler
    """
    categorical_cols = ['Income_type', 'Education_type', 'Family_status', 'Housing_type', 'Occupation_type']
    numeric_cols = ['Total_income', 'Age', 'Years_employed', 'Account_length']
    
    # One-Hot Encoding
    encoder = OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')
    encoded_cols = pd.DataFrame(encoder.fit_transform(data[categorical_cols]))
    encoded_cols.columns = encoder.get_feature_names_out(categorical_cols)
    encoded_cols.index = data.index
    
    # Drop original categorical columns and merge encoded ones
    data = data.drop(columns=categorical_cols).join(encoded_cols)
    
    # Fill missing values with median
    data = data.fillna(data.median())
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    data[numeric_cols] = scaler.fit_transform(data[numeric_cols])
    
    return data

# Endpoint to receive data from a form
@app.post("/predict")
def predict(data: InputData):
    """Receive JSON input, preprocess it, and return a binary prediction."""
    df = pd.DataFrame([data.dict()])
    df = preprocess(df)
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}

# Endpoint to receive a CSV file from WordPress
@app.post("/predict_csv")
def predict_csv(file: UploadFile = File(...)):
    """Receive CSV file, preprocess it, and return predictions for each row."""
    contents = file.file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    df = preprocess(df)
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
