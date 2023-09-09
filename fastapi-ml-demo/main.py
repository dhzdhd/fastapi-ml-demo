from typing import Union
from pydantic import BaseModel

from fastapi import FastAPI
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
import pickle

model = tf.keras.models.load_model("fastapi-ml-demo/notebooks/model.keras")
oe: OrdinalEncoder | None = None

with open("fastapi-ml-demo/notebooks/encoder.p", "rb") as f:
    oe = pickle.load(f)

app = FastAPI()

class InputModel(BaseModel):
    PassengerId: str
    HomePlanet: str
    CryoSleep: bool
    Cabin: str
    Destination: str
    Age: float
    VIP: bool
    RoomService: float
    FoodCourt: float
    ShoppingMall: float
    Spa: float
    VRDeck: float
    Name: str

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def post_predict(input: InputModel) -> dict[str, bool]:
    df = pd.DataFrame(dict(input), index=[0])
    df[["HomePlanet", "CryoSleep", "Cabin", "Destination", "VIP"]] = oe.fit_transform(df[["HomePlanet", "CryoSleep", "Cabin", "Destination", "VIP"]])
    df = df.drop(["PassengerId", "Name"], axis=1)

    print(df)
    y_pred = model.predict(df)
    result = np.where(y_pred > 0.5, True, False).flatten()

    return { "Prediction": result[0] }
