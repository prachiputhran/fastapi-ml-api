from fastapi import FastAPI
import pickle
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define input format (JSON)
class InputData(BaseModel):
    features: list

@app.get("/")
def home():
    return {"message": "FastAPI ML API is running"}

@app.post("/predict")
def predict(data: InputData):
    arr = np.array(data.features).reshape(1, -1)
    prediction = model.predict(arr)
    return {"prediction": int(prediction[0])}

@app.get("/")
def home():
    return {"message": "FastAPI ML API is running"}