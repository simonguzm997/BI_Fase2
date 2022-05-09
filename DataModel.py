# Librerias
from pydantic import BaseModel
from typing import List

# Clase DataModel
class DataModel(BaseModel):

    label: str
    study_and_condition: str

    # Columbas usadas 
    def columns():
        return ["label", "study_and_condition"]

class DataList(BaseModel):
    data: List[DataModel]
