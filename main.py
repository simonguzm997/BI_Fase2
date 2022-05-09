# Librerias
import json
from plistlib import load
from DataModel import DataModel, DataList, DMpredictVar
from pandas import json_normalize
from sklearn.metrics import r2_score
from joblib import load
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI
import os


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
    modelo = load("assets/modelo.joblib")
    resultado = modelo.predict(df)
    lista = resultado.tolist()
    json_predict = json.dumps(lista)
    return {"predict": json_predict}

@app.post("/r2_model")
def r2_model(data: DataList, dataTrue: DMpredictVar):
    df = json_converter(data)
    df.columns = DataModel.columns()
    modelo = load("assets/modelo.joblib")
    resultado = modelo.predict(df)
    dic = jsonable_encoder(dataTrue)
    y_true = []
    for i in dic["dataTrue"]:
        y_true.append(float(i["life_expectancy"]))
    r2 = r2_score(y_true, resultado.tolist())
    return {"r^2": r2}

# Convertidor de json a data frame
def json_converter(data):
    dict = jsonable_encoder(data)
    dataframe = json_normalize(dict['data']) 
    return dataframe