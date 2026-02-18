from fastapi import FastAPI
import joblib
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load model once when API starts
model_data = joblib.load("churn_model.pkl")
model = model_data["model"]
threshold = model_data["threshold"]


@app.get("/")
def home():
    return {"message": "Churn Prediction API is running"}

from pydantic import BaseModel

class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.post("/predict")
def predict(data: CustomerData):

    input_df = pd.DataFrame([data.dict()])

    # Generate ChargesPerTenure (IMPORTANT — matches training)
    input_df["ChargesPerTenure"] = input_df["MonthlyCharges"] / (input_df["tenure"] + 1)

    # Get probability
    proba = model.predict_proba(input_df)[0][1]

    # Apply threshold
    pred = 1 if proba >= threshold else 0

    prediction_label = "Churn" if pred == 1 else "No Churn"

    return {
        "prediction": prediction_label,
        "probability": round(float(proba), 4)
    }

