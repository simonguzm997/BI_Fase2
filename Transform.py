#!/usr/bin/env python
# coding: utf-8

# # Proyecto 

# ### 1. Perfilación y preparación
from IPython import embed
# En las siguientes líneas de código se importan las librerías y herramientas necesarias para desarrollar el caso de uso.

# In[1]:


# Librería para manejar las contracciones que se presentan en el inglés.
get_ipython().system('pip install contractions')


# In[2]:


# librería para manejar las flexiones gramaticales en el idioma inglés.
get_ipython().system('pip install inflect')
get_ipython().system('pip install pandas-profiling==2.7.1')


# In[3]:


# librería Natural Language Toolkit, usada para trabajar con textos 
import nltk
# Punkt permite separar un texto en frases.
nltk.download('punkt')


# In[4]:


# Descarga todas las palabras vacias, es decir, aquellas que no aportan nada al significado del texto
# ¿Cuales son esas palabras vacías?

nltk.download('stopwords')


# In[5]:


# Descarga de paquete WordNetLemmatizer, este es usado para encontrar el lema de cada palabra
# ¿Qué es el lema de una palabra? ¿Qué tan dificil puede ser obtenerlo, piensa en el caso en que tuvieras que escribir la función que realiza esta tarea?
nltk.download('wordnet')


# In[30]:


# Instalación de librerias
import pandas as pd
import numpy as np
import sys
import seaborn as sns
from pandas_profiling import ProfileReport

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
get_ipython().run_line_magic('matplotlib', 'inline')


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from joblib import dump, load




# Carga de los datos

class Transofrm(DataModel):

	data=DataModel

# Se cargan los datos.

# In[10]:


# Cantidad de datos y número de variables
	data.shape


# In[11]:


# Mostrar los datos
	print(data.head())
	

# In[12]:


# Es recomendable que todos los pasos preparación se realicen sobre otro archivo.
	data_t = data


# In[13]:


	data_t.info()


# In[14]:


	data_t['label'].value_counts()


# ### Limpieza de los datos
# Para dejar el archivo en texto plano, sobre todo cuando vienen de diferentes fuentes como HTML, Twitter, XML, entre otros. También para eliminar caracteres especiales y pasar todo a minúscula.

# In[15]:
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
    """Replace all interger occurrences in list of tokenized words with textual representation"""
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
    """Remove stop words from list of tokenized words"""
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




# In[16]:


# Eliminación registros con ausencias
	data_t = data_t.dropna()
# Eliminación de registros duplicados.
	data_t = data_t.drop_duplicates()	
	data_t['label'].value_counts()


# ### Tokenización
# La tokenización permite dividir frases u oraciones en palabras. Con el fin de desglozar las palabras correctamente para el posterior análisis. Pero primero, se realiza una corrección de las contracciones que pueden estar presentes en los textos. 

# In[17]:


	data_t['study_and_condition'] = data_t['study_and_condition'].apply(contractions.fix) #Aplica la corrección de las contracciones


# In[18]:


	data_t['words'] = data_t['study_and_condition'].apply(word_tokenize).apply(preprocessing) #Aplica la eliminación del ruido
	data_t.head()


# Eliminar palabras repetidas en todos los registros (study, interventions)

# In[19]:


	new_words = []
	for word in data_t['words']:
    	new_words = word.remove('study')
    	new_words = word.remove('interventions')
    	data_t['words'] = data_t['words'].replace(new_words)
	data_t.head()


# ### Normalización
# En la normalización de los datos se realiza la eliminación de prefijos y sufijos, además de realizar una lemmatización.

# In[20]:


	lemmatizer = nltk.stem.WordNetLemmatizer()
	wordnet_lemmatizer = WordNetLemmatizer()
	stop = stopwords.words('english')

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
    """Stem words in list of tokenized words"""
    #https://www.datacamp.com/community/tutorials/stemming-lemmatization-python
    	porter = PorterStemmer()
    	lancaster=LancasterStemmer()
    	new_words = []
    	for word in words:
        	new_words.append(porter.stem(word))
    	return new_words
        

	def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    #https://www.geeksforgeeks.org/python-lemmatization-approaches-with-examples/
    	wnl = WordNetLemmatizer()
    	new_words = []
    	for word in words:
        	new_words.append(wnl.lemmatize(word))
    	return new_words


	def stem_and_lemmatize(words):
    	stems = stem_words(words)
    	lemmas = lemmatize_verbs(words)
    	return stems + lemmas


	data_t['words'] = data_t['words'].apply(stem_and_lemmatize) #Aplica lematización y Eliminación de Prefijos y Sufijos.
	data_t.head()


# ###  Selección de campos
# 
# Primero, se separa la variable predictora y los textos que se van a utilizar.

# In[21]:


	data_t['words'] = data_t['words'].apply(lambda x: ' '.join(map(str, x)))
	


# In[22]:


	data_t['label'] = data_t['label'].replace(['__label__1'],1)
	data_t['label'] = data_t['label'].replace(['__label__0'],0)


# Aplicamos TF_IDF (Term-frecuency times inverse Document-frecuency) a los datos
# 
# 
# 
# 

# In[23]:


# Source: https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76
	vectorizer = TfidfVectorizer()
	allDocs = []
	for word in data_t['words']:
    	allDocs.append(word)
	vectors = vectorizer.fit_transform(allDocs)
	feature_names = vectorizer.get_feature_names()
	dense = vectors.todense()
	denselist = dense.tolist()
	data_tfidf = pd.DataFrame(denselist, columns=feature_names)
	print(data_tfidf.head())


	return data_tfidf



