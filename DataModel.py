# Librerias
from pydantic import BaseModel
from typing import List

# Clase DataModel
class DataModel(BaseModel):

    label: object
    study_and_condition: object


    # Columbas usadas 
    def columns():
        return ["label", "study_and_condition"]

class DataList(BaseModel):
    data: List[DataModel]

class datMod(BaseModel):
    life_expectancy: float
    def column():
        return "Life expectancy"
class DMpredictVar(BaseModel):
    dataTrue: List[datMod]
