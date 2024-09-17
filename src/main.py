# main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.model import predict_churn
from src.explanations import global_explanation, local_explanation, surrogate_model_explanation

app = FastAPI()

class Features(BaseModel):
    features: dict
    instance_idx: int = None  

@app.post("/predict/")
async def predict(features: Features):
    try:
        df = pd.DataFrame([features.features])
        prediction = predict_churn(df)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/explain/global/")
async def explain_global():
    try:
        df = pd.DataFrame(r'C:\Users\ADMIN\OneDrive\Desktop\TASK\survey_sparrow\data\churn_data\Bank_churn.csv')  
        explanation = global_explanation(df)
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/explain/local/")
async def explain_local(features: Features):
    try:
        df = pd.DataFrame([features.features])
        explanation = local_explanation(df, features.instance_idx)
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/explain/surrogate/")
async def explain_surrogate():
    try:
        df = pd.DataFrame(r'C:\Users\ADMIN\OneDrive\Desktop\TASK\survey_sparrow\data\churn_data\Bank_churn.csv')  
        surrogate = surrogate_model_explanation(df)
        return {"surrogate_model": surrogate}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
