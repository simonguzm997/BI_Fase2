# Librerias
import json
from plistlib import load
from DataModel import DataModel, DataList
from pandas import json_normalize
from sklearn.metrics import r2_score
from joblib import load
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI
import os
import Transform

# Instancia de FastAPI
app = FastAPI()
cmd = "ipython pipeline_rl.py"
os.system(cmd)

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def make_predictions(data: DataList):
    df = json_converter(data)
    df.columns = DataModel.columns()
    
    data=Transform(df)
    
    modelo = load("assets/modelo.joblib")
    resultado = modelo.predict(data)
    lista = resultado.tolist()
    json_predict = json.dumps(lista)
    return {"Prediction": json_predict}

# Convertidor de json a data frame
def json_converter(data):
    dict = jsonable_encoder(data)
    dataframe = json_normalize(dict['data']) 
    return dataframe