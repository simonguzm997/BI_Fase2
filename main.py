# Librerias
import json
from datetime import datetime
from plistlib import load
from DataModel import DataModel
from pandas import json_normalize
from joblib import load
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI
import os
import pandas as pd
import numpy as np
import sys
import seaborn as sns
from pandas_profiling import ProfileReport
import nltk
import nltk; nltk.download('omw-1.4')
import nltk; nltk.download('punkt')
import nltk; nltk.download('stopwords')
import nltk; nltk.download('wordnet')
import re, string, unicodedata
import contractions
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from pandas.core.dtypes.generic import ABCIndex
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, plot_precision_recall_curve
from sklearn.base import BaseEstimator, ClassifierMixin
from statistics import mode
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import validation_curve
# Para búsqueda de hiperparámetros
from sklearn.model_selection import GridSearchCV
# Para la validación cruzada
from sklearn.model_selection import KFold 
# Para usar KNN como clasificador
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# Para la regresion logistica
from sklearn import linear_model
from sklearn import model_selection

# Instancia de FastAPI
app = FastAPI()
cmd = "ipython pipeline_rl.py"
#os.system(cmd)


now = datetime.now()

current_time = now.strftime("%H:%M:%S")

print("\n\n * Inicializar API @", current_time, "* \n")


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/predict")
def make_prediction(data: DataModel):
    df = transformer(data)
    #df.columns = DataModel.columns()
    modelo = load("assets/modelo.joblib")
    resultado = modelo.predict(df)
    #lista = resultado.tolist()
    #json_predict = json.dumps(lista)
    return {"Prediccion": resultado}

# Convertidor de json a data frame
def transformer(data):
    #dict = jsonable_encoder(data)
    #data_t = json_normalize(dict['data'])
    data_t = pd.DataFrame(data.dict(), columns=data.dict().keys(), index=[0])
    print("\n DATAFRAME ORIGINAL: ")
    print(data_t.head())
    
    data_t['study_and_condition'] = data_t['study_and_condition'].apply(contractions.fix)
    data_t['words'] = data_t['study_and_condition'].apply(word_tokenize).apply(preprocessing)
    
    new_words = []
    for word in data_t['words']:
    	new_words=word.remove('study')
    	new_words=word.remove('interventions')
    	data_t['words']=data_t['words'].replace(new_words)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    wordnet_lemmatizer = WordNetLemmatizer
    stop = stopwords.words('english')
    data_t['words'] = data_t['words'].apply(stem_and_lemmatize)
    data_t['words'] = data_t['words'].apply(lambda x: ' '.join(map(str, x)))
    #data_t['label'] = data_t['label'].replace(['__label__1'],1)
    #data_t['label'] = data_t['label'].replace(['__label__0'],0)
    vectorizer = TfidfVectorizer()
    allDocs = []
    for word in data_t['words']:
    	allDocs.append(word)
    vectors = vectorizer.fit_transform(allDocs)
    feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    data_tfidf = pd.DataFrame(denselist, columns=feature_names)
    df_limpio = data_tfidf
    print("\n DATAFRAME LIMPIO: ")
    print(df_limpio.head())
    return df_limpio


#DEF Limpieza

  
def remove_non_ascii(words):
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words
    
def to_lowercase(words):
    new_words = []
    for word in words:
        new_words.append(word.lower())
    return new_words
    
def remove_punctuation(words):
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words
    
def replace_numbers(words):
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words
    
def remove_stopwords(words):
    new_words = []
    stop = stopwords.words('english')
    
    for word in words:
        if word not in (stop):
            new_words.append(word)
    return new_words
    
def preprocessing(words):
    words = to_lowercase(words)
    words = replace_numbers(words)
    words = remove_punctuation(words)
    words = remove_non_ascii(words)
    words = remove_stopwords(words)
    return words
    
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return wordnet.ADJ
    elif nltk_tag.startswith('V'):
        return wordnet.VERB
    elif nltk_tag.startswith('N'):
        return wordnet.NOUN
    elif nltk_tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def stem_words(words):
    porter = PorterStemmer()
    lancaster=LancasterStemmer()
    new_words = []
    for word in words:
        new_words.append(porter.stem(word))
    return new_words

def lemmatize_verbs(words):
    wnl = WordNetLemmatizer()
    new_words = []
    for word in words:
        new_words.append(wnl.lemmatize(word))
    return new_words   

def stem_and_lemmatize(words):
    stems = stem_words(words)
    lemmas = lemmatize_verbs(words)
    return stems + lemmas   
    

print("\n * Inicializacion Completa * \n")
