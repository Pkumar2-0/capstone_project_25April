from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load model + encoders
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")

API_KEY = "secure123"

# Input schema
class InputData(BaseModel):
    product_id: int
    region: str
    date: str   # optional (not used unless trained)

@app.post("/predict-demand")
def predict(data: InputData, x_api_key: str = Header(...)):
    
    # Security check
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

    input_dict = data.dict()

    # Encode categorical values
    for col in encoders:
        if col in input_dict:
            input_dict[col] = encoders[col].transform([input_dict[col]])[0]

    input_array = np.array([list(input_dict.values())])

    prediction = model.predict(input_array)

    return {"predicted_demand": float(prediction[0])}