# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import json
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

def edad(birthdate):
    return int(((datetime.now() - birthdate).days)/365.2425)

with open('2_Inputs/deliveryDatasetChallenge.json') as json_file:
    data = json.load(json_file)
    
    print("Type:", type(data))


data = pd.DataFrame(data)
data_columns = data.columns.str.split(';',expand=True)

data = data.iloc[:,0].str.split(';',expand=True)

data.columns = ['anonID','birthdate','routeDate','region','gender','areaWealthLevel','areaPopulation',
                'badWeather','weatherRestrictions','routeTotalDistance','numberOfShops','marketShare',
                'avgAreaBenefits','timeFromAvg','advertising','employeeLYScore','employeeTenure','employeePrevComps','success']

data = data.reset_index().drop(["index"], axis =1)

del json_file
del data_columns

#####################
#Formato de columnas#
#####################
#Variables númericas : 'areaPopulation','routeTotalDistance','numberOfShops','marketShare', 'avgAreaBenefits', 'timeFromAvg',
#                      'advertising', 'employeeLYScore','employeeTenure', 'employeePrevComps'
data["areaPopulation"] = data["areaPopulation"].astype(float)
data['routeTotalDistance'] = data['routeTotalDistance'].astype(float)
data['numberOfShops'] = data['numberOfShops'].astype(int)
data['marketShare'] = data['marketShare'].astype(float)

data.loc[data['avgAreaBenefits'] == 'NA','avgAreaBenefits'] = 0
data['avgAreaBenefits'] = data['avgAreaBenefits'].astype(float)

data.loc[data['timeFromAvg'] == 'NA','timeFromAvg'] = 0
data['timeFromAvg'] = data['timeFromAvg'].astype(float)

data['advertising'] = data['advertising'].astype(int)
data['employeeLYScore'] = data['employeeLYScore'].astype(int)
data['employeeTenure'] = data['employeeTenure'].astype(int)
data['employeePrevComps'] = data['employeePrevComps'].astype(int)

#Variables categorícas : 'anonID', 'region', 'gender', 'areaWealthLevel','badWeather', 'weatherRestrictions','success'
data.loc[data['success'] == 'NA','success'] = -1
data['success'] = data['success'].astype(int)
data["region-gender-areaWealthLevel-badWeather-weatherRestrictions"] = data["region"] +"-"+ data["gender"] +"-"+ data["areaWealthLevel"] +"-"+ data["badWeather"] +"-"+ data["badWeather"] 

#Variables tipo fecha
data["birthdate"] = pd.to_datetime(data["birthdate"])
data["edad"] = data["birthdate"].apply(edad)

data["routeDate"] = np.where(data["routeDate"] == 'NA', '', data["routeDate"])
data["routeDate"] = pd.to_datetime(data["routeDate"])
data["month"] = data["routeDate"].dt.month.astype(str)
data["day"] = data["routeDate"].dt.day.astype(str)
data["year"] = data["routeDate"].dt.year.astype(str)
data["month"] = data["month"].apply(lambda x : x.replace('.0',''))
data["day"] = data["day"].apply(lambda x : x.replace('.0',''))
data["year"] = data["year"].apply(lambda x : x.replace('.0',''))

#Reacomodo
data = data.loc[:,["anonID","birthdate","edad","routeDate","month","day","year","region","gender","areaWealthLevel","badWeather","weatherRestrictions",
                   "region-gender-areaWealthLevel-badWeather-weatherRestrictions","areaPopulation","routeTotalDistance","numberOfShops","marketShare",
                   "avgAreaBenefits","timeFromAvg","advertising","employeeLYScore","employeeTenure","employeePrevComps","success"]]

######################
#Exploración de datos#
######################
data_filter = data[(data["success"] >= 0) & ~(data["routeDate"].isnull())]

#Variables categoricas, respecto a la distribución (0,1)
def grupos(grupo = ['region','success'], num_grupos = 2):
    if num_grupos == 1:
        return data_filter.groupby(grupo).count().iloc[:,:1]
    else :
        return data_filter.groupby(grupo).agg({'success': 'count'}).groupby(level=0).apply(lambda x: round(100 * x / float(x.sum()),1)) 

grupos(['success'],num_grupos=1)
grupos(['routeDate','success'])
grupos(['region','success'])
grupos(['gender','success'])
grupos(['areaWealthLevel','success'])
grupos(['badWeather','success'])
grupos(['weatherRestrictions','success'])
tmp = grupos(['region-gender-areaWealthLevel-badWeather-weatherRestrictions','success'])
          
#Distribución variables númericas
data_filter['areaPopulation'].describe() # Población de la zona cubierta, en miles
data_filter['routeTotalDistance'].describe() #Distancia de la ruta recorrida en kms

data_filter['numberOfShops'].describe() #Total Tiendas que cubrimos en la zona
data_filter['marketShare'].describe() #Porcentaje de cuota de mercado que la empresa tiene en la zona en sus categorías.
data_filter['avgAreaBenefits'].describe() #Beneficio económico semanal en la zona (en miles de $)
data_filter['timeFromAvg'].describe() #Tiempo empleado en la ruta, comparado con la media (negativo significaría que se tardó menos que la media)

#Distribución variables discretas
grupos(['edad','success']) #No hay tanta variedad de resultados por edad
grupos(['advertising','success']) #Inversión en material de punto de venta en las tiendas (de 0, que significa que no se invierte, a 3, que se invierte mucho)
grupos(['employeeLYScore','success']) #Calificando la puntuación del año pasado (de 1 a 5, siendo 5 la más alta). Los nuevos empleados tienen 3 por defecto.
grupos(['employeeTenure','success']) #Tiempo que el empleado lleva en la empresa
grupos(['employeePrevComps','success']) #Número de empresas en las que el empleado trabajó anteriormente desarrollando la misma función (5 significa 3 o más).

#################
#Modelo de datos#
#################
from sklearn.preprocessing import LabelEncoder

data_select = data_filter.loc[:,["edad","region","gender","areaWealthLevel",
                                 "badWeather", "weatherRestrictions", "areaPopulation", 
                                 "routeTotalDistance","numberOfShops","marketShare","avgAreaBenefits",
                                 "timeFromAvg","advertising","employeeLYScore","employeeTenure","employeePrevComps","success"]]

le = LabelEncoder()
data_select["region"] = le.fit_transform(data_select["region"])
data_select["gender"] = le.fit_transform(data_select["gender"])
data_select["areaWealthLevel"] = le.fit_transform(data_select["areaWealthLevel"])
data_select["badWeather"] = le.fit_transform(data_select["badWeather"])
data_select["weatherRestrictions"] = le.fit_transform(data_select["weatherRestrictions"])

#Tenemos 5 variables más o menos correlacionadas positivamente
#advertising ~ employeeLYScore     0.615878 => Entre mayor es la inversión mayor es la calificación
#employeeLYScore ~ areaPopulation     0.594624 
#badWeather ~ weatherRestrictions     0.408545 => Mal clima implica afectaciones en la zona.
#employeeTenure ~ employeePrevComps .35 => En ocasiones entre más tiempo tiene en la empresa ya estuvo en otras empresas.
#advertising ~ avgAreaBenefits     0.33 => En ocasiones entre mayor es la inversión mejores beneficios en la zona.
 
tmp = data_select.corr().reset_index()
tmp = tmp.melt(id_vars = 'index', value_vars =['edad', 'region', 'gender', 'areaWealthLevel', 'badWeather',
       'weatherRestrictions', 'areaPopulation', 'routeTotalDistance',
       'numberOfShops', 'marketShare', 'avgAreaBenefits', 'timeFromAvg',
       'advertising', 'employeeLYScore', 'employeeTenure',
       'employeePrevComps'], 
        var_name ='variable_corr', value_name ='corr_person').sort_values("corr_person",ascending = False)
tmp = tmp[tmp["corr_person"] != 1]
tmp.head(25)
tmp.tail()

#Propuesta Modelo : Supervisado -> Clasificación : (KNeighborsClassifier, LogisticRegression, LinearSVC, SVC)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

#Modelo KNN
data_model = data_select
x, y = data_model.iloc[:,:-1], data_model.success

xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.3, random_state = 21, stratify = y)

ytest.value_counts(normalize = True)
ytrain.value_counts(normalize = True)

knc = KNeighborsClassifier(n_neighbors=3)
knc.fit(xtrain, ytrain)
score = knc.score(xtrain, ytrain)
print("Score: ", score)

ypred = knc.predict(xtest)

#Métricas de Clásificación.
#     Predicted: 0   Predicted: 1
#0    True Positive  False Negative
#1    False Positive True Negative

cm = confusion_matrix(ytest, ypred)
ytest.value_counts()
print(cm)

accuracy_score(ytest, ypred) #Porcentaje total de valores correctamente clasificados, tanto positivos como negativos.
precision_score(ytest, ypred) #Saber qué porcentaje de valores que se han clasificado como positivos son realmente positivos.
recall_score(ytest, ypred) #Cuantos valores positivos son correctamente clasificados.
f1_score(ytest, ypred) #

cr = classification_report(ytest, ypred)
print(cr)
  
#Curva Roc
fpr, tpr, threshold = roc_curve(ytest, ypred)
roc_auc = auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('ROC Curve of kNN')
plt.show()

#Tuneo de hyperparametros.

#Regiones de desición variando los parametros
plt.rcParams['figure.figsize'] = [20, 20]
plt.title("Knn región de desición líneal")
plt_cls.plot_decision_boundaries(X, y,  KNeighborsClassifier,n_neighbors=2)
plt_cls.plot_decision_boundaries(X, y,  KNeighborsClassifier,n_neighbors=8)
plt_cls.plot_decision_boundaries(X, y,  KNeighborsClassifier,n_neighbors=15)

#Grafica variando el numero de n_neighbors y su presición.
n_neighbors = range(1,20)
train  = []
test = []

for i in n_neighbors:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    
    train.append(knn.score(X_train, y_train))
    test.append(knn.score(X_test, y_test))

plt.rcParams['figure.figsize'] = [5, 5]
line_up, = plt.plot(n_neighbors, train , label='Training Accuracy')
line_down, = plt.plot(n_neighbors, test, label='Testing Accuracy')
plt.title("KNN : Variación de numero de cluster")
plt.legend([line_up, line_down], ['Training Accuracy', 'Testing Accuracy'])
plt.annotate('Overfiting', xy = (1.5, .94), arrowprops = {'facecolor':'gray', 'width': 3, 'shrink': 0.03})
plt.annotate('Optimo', xy = (8, .94), arrowprops = {'facecolor':'gray', 'width': 3, 'shrink': 0.03})
plt.annotate('Underfiting', xy = (16, .94), arrowprops = {'facecolor':'gray', 'width': 3, 'shrink': 0.03})

#Sesgo y Varianza.
#Dejar modularizado el código.
#Crear data set de resultados.
#Probar otro modelo.

#Seguir exploración de variables.
