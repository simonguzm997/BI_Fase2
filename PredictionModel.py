# Librerias
from joblib import load

# Model class
class Model:

    # Constructor
    def __init__(self,columns):
        self.model = load("assets/modelo.joblib")

    # Predect
    def make_predictions(self, data):
        resultados = self.model.predict(data)
        return resultados