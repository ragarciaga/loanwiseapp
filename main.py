from fastapi import FastAPI, File, UploadFile
import pandas as pd
import pickle
import uvicorn
from pydantic import BaseModel
import io
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from fastapi.middleware.cors import CORSMiddleware

# ✅ Configurar CORS para permitir solicitudes desde WordPress
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://caribbeanspice.musicpro.online"],  # Dominio de WordPress
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Permite todos los encabezados
)

# ✅ Cargar el modelo entrenado
with open("model_experiment_9.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ Cargar el encoder, scaler y columnas en el orden correcto
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("column_order.pkl", "rb") as f:
    column_order = pickle.load(f)

# ✅ Definir la estructura de entrada de datos
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

# ✅ Función para preprocesar los datos antes de hacer la predicción
def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess input data before making predictions:
    - Apply One-Hot Encoding using the pre-trained encoder
    - Fill missing values with median
    - Normalize numerical features using the pre-trained MinMaxScaler
    - Ensure correct column order
    """
    categorical_cols = ['Income_type', 'Education_type', 'Family_status', 'Housing_type', 'Occupation_type']
    numeric_cols = ['Total_income', 'Age', 'Years_employed', 'Account_length']

    # One-Hot Encoding using the pre-trained encoder
    encoded_cols = pd.DataFrame(encoder.transform(data[categorical_cols]))
    encoded_cols.columns = encoder.get_feature_names_out(categorical_cols)
    encoded_cols.index = data.index

    # Drop original categorical columns and merge encoded ones
    data = data.drop(columns=categorical_cols).join(encoded_cols)

    # Fill missing values with median
    data = data.fillna(data.median())

    # Normalize numerical features using the pre-trained scaler
    data[numeric_cols] = scaler.transform(data[numeric_cols])

    # Asegurar que las columnas están en el mismo orden que en el entrenamiento
    for col in column_order:
        if col not in data.columns:
            data[col] = 0  # Agregar columnas faltantes con valores 0

    data = data[column_order]  # Reordenar las columnas según el entrenamiento

    return data

# ✅ Endpoint para recibir datos JSON desde WordPress
@app.post("/predict")
def predict(data: InputData):
    """Receive JSON input, preprocess it, and return a binary prediction."""
    df = pd.DataFrame([data.dict()])
    df = preprocess(df)
    prediction = model.predict(df)
    return {"prediction": int(prediction[0])}

# ✅ Endpoint para recibir archivos CSV desde WordPress
@app.post("/predict_csv")
def predict_csv(file: UploadFile = File(...)):
    """Receive CSV file, preprocess it, and return predictions for each row."""
    contents = file.file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    df = preprocess(df)
    predictions = model.predict(df)
    return {"predictions": predictions.tolist()}

# ✅ Ejecutar la API en el puerto 8000
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
