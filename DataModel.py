# Librerias
from pydantic import BaseModel
from typing import List
import string 

# Clase DataModel
class DataModel(BaseModel):

    study_and_condition: object

    # Columbas usadas 
    def columns():
        return ["study_and_condition"]
        


