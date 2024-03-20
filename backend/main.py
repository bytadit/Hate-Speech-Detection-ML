import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from model.predict import predictNewData

api = FastAPI()

class TextItem(BaseModel):
    text: str

@api.get('/')
async def api_home():
    return {"greeting": "Welcome to HS Api!"}

@api.post('/predict')
async def predict_text(item:TextItem):
    return predictNewData(item.text)