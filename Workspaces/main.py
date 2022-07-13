# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 13:30:27 2022

@author: cflorelu
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio

from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from mlxtend.evaluate import bias_variance_decomp
from sklearn.metrics import zero_one_loss

from Utils import preparacionDatos
from Utils import mlDatos

with open('Inputs/deliveryDatasetChallenge.json') as json_file:
    data = json.load(json_file)
    
    print("Type:", type(data))


data = pd.DataFrame(data)

data = preparacionDatos.dataClean(data)

data_filter = preparacionDatos.dataFilter(data)

data_model = preparacionDatos.dataModel(data_filter)

######################
#Exploración de datos#
######################
preparacionDatos.grupos(data_filter,['success'],num_grupos=1)
preparacionDatos.grupos(data_filter,['routeDate','success'])
preparacionDatos.grupos(data_filter,['region','success'])
preparacionDatos.grupos(data_filter,['gender','success'])
preparacionDatos.grupos(data_filter,['areaWealthLevel','success'])
preparacionDatos.grupos(data_filter,['badWeather','success'])
preparacionDatos.grupos(data_filter,['weatherRestrictions','success'])
tmp = preparacionDatos.grupos(data_filter,['region-gender-areaWealthLevel-badWeather-weatherRestrictions','success'])
          
#Distribución variables númericas
data_filter['areaPopulation'].describe() # Población de la zona cubierta, en miles
data_filter['routeTotalDistance'].describe() #Distancia de la ruta recorrida en kms

data_filter['numberOfShops'].describe() #Total Tiendas que cubrimos en la zona
data_filter['marketShare'].describe() #Porcentaje de cuota de mercado que la empresa tiene en la zona en sus categorías.
data_filter['avgAreaBenefits'].describe() #Beneficio económico semanal en la zona (en miles de $)
data_filter['timeFromAvg'].describe() #Tiempo empleado en la ruta, comparado con la media (negativo significaría que se tardó menos que la media)

#Distribución variables discretas
preparacionDatos.grupos(data_filter,['edad','success']) #No hay tanta variedad de resultados por edad
preparacionDatos.grupos(data_filter,['advertising','success']) #Inversión en material de punto de venta en las tiendas (de 0, que significa que no se invierte, a 3, que se invierte mucho)
preparacionDatos.grupos(data_filter,['employeeLYScore','success']) #Calificando la puntuación del año pasado (de 1 a 5, siendo 5 la más alta). Los nuevos empleados tienen 3 por defecto.
preparacionDatos.grupos(data_filter,['employeeTenure','success']) #Tiempo que el empleado lleva en la empresa
preparacionDatos.grupos(data_filter,['employeePrevComps','success']) #Número de empresas en las que el empleado trabajó anteriormente desarrollando la misma función (5 significa 3 o más).

#Analisis data model correlaciones.
#Tenemos 5 variables más o menos correlacionadas positivamente
#advertising ~ employeeLYScore     0.615878 => Entre mayor es la inversión mayor es la calificación
#employeeLYScore ~ areaPopulation     0.594624 
#badWeather ~ weatherRestrictions     0.408545 => Mal clima implica afectaciones en la zona.
#employeeTenure ~ employeePrevComps .35 => En ocasiones entre más tiempo tiene en la empresa ya estuvo en otras empresas.
#advertising ~ avgAreaBenefits     0.33 => En ocasiones entre mayor es la inversión mejores beneficios en la zona.
 
tmp = data_model.corr().reset_index()
tmp = tmp.melt(id_vars = 'index', value_vars =['edad', 'region', 'gender', 'areaWealthLevel', 'badWeather',
       'weatherRestrictions', 'areaPopulation', 'routeTotalDistance',
       'numberOfShops', 'marketShare', 'avgAreaBenefits', 'timeFromAvg',
       'advertising', 'employeeLYScore', 'employeeTenure',
       'employeePrevComps'], 
        var_name ='variable_corr', value_name ='corr_person').sort_values("corr_person",ascending = False)
tmp = tmp[tmp["corr_person"] != 1]
tmp.head(25)
tmp.tail()

#################################
#Primer Modelo Clasificación KNN#
#################################
#Propuesta Modelo : Supervisado -> Clasificación : (KNeighborsClassifier, LogisticRegression, LinearSVC, SVC)

#Modelo KNN
x, y = data_model.iloc[:,:-1], data_model.success

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 123, stratify = y)

y_test.value_counts(normalize = True)
y_train.value_counts(normalize = True)

#Grafica variando el numero de n_neighbors y su presición.
n_neighbors = range(1,7)
train  = []
test = []

for i in n_neighbors:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    
    train.append(knn.score(x_train, y_train))
    test.append(knn.score(x_test, y_test))

plt.rcParams['figure.figsize'] = [5, 5]
line_up, = plt.plot(n_neighbors, train , label='Entrenamiento Accuracy')
line_down, = plt.plot(n_neighbors, test, label='Testeo Accuracy')
plt.title("KNN : Variación de numero de cluster.")
plt.legend([line_up, line_down], ['Entrenamiento Accuracy', 'Testeo Accuracy'])

plt.annotate('Overfiting: Bajo sesgo, alta varianza.', xy = (2, .982), arrowprops = {'facecolor':'gray', 'width': 3, 'shrink': 0.03})
plt.annotate('Optimo', xy = (4, .95), arrowprops = {'facecolor':'gray', 'width': 3, 'shrink': 0.03})
plt.annotate('Underfiting : Alto sesgo, baja varianza.', xy = (2, .93), arrowprops = {'facecolor':'gray', 'width': 3, 'shrink': 0.03})

#Para varios valores de k en kNN, examinemos también cómo van a ser nuestras pérdidas, sesgos y varianzas
bias_KnnClass, var_KnnClass,error_KnnClass, = [], [], []
for k in range(1,8):
    clf_knn = KNeighborsClassifier(n_neighbors=k)
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(clf_knn, x_train.values, y_train.values, x_test.values, y_test.values, loss = '0-1_loss', random_seed = 123)
    bias_KnnClass.append(avg_bias)
    var_KnnClass.append(avg_var)
    error_KnnClass.append(avg_expected_loss)
    print(f"Pérdida media esperada {avg_expected_loss}")
    print(f"Media bias {avg_bias}")
    print(f"Media variance {avg_var}")
plt.plot(range(1,8), error_KnnClass, 'red', label = 'total_error',linestyle='dashed')
plt.plot(range(1,8), bias_KnnClass, 'brown', label = 'bias^2')
plt.plot(range(1,8), var_KnnClass, 'yellow', label = 'varianza')
plt.xlabel('Complejidad del modelo con (K)')
plt.ylabel('Error')
plt.legend()

#Mejor modelo analizando knn con Sesgo y Varianza.
clf_knn = KNeighborsClassifier(n_neighbors=5)
clf_knn.fit(x_train, y_train)
score = clf_knn.score(x_train, y_train)
print("Score: ", score)

y_pred = clf_knn.predict(x_test)

avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(clf_knn, x_train.values, y_train.values, x_test.values, y_test.values, loss = '0-1_loss', random_seed = 123)
print('Pérdida media esperada: %.3f' % avg_expected_loss)
print('Media Bias: %.3f' % avg_bias)
print('Media Variance: %.3f' % avg_var)
print('Skelarn 0-1 perdida : %.3f' % zero_one_loss(y_test, y_pred))

#Métricas de Clásificación para nuestro modelo knn = 4.
#     Predicted: 0   Predicted: 1
#0    True Positive  False Negative
#1    False Positive True Negative

cm = confusion_matrix(y_test, y_pred)
y_test.value_counts()
print(cm)

accuracy_score(y_test, y_pred) #Porcentaje total de valores correctamente clasificados, tanto positivos como negativos.
precision_score(y_test, y_pred) #Saber qué porcentaje de valores que se han clasificado como positivos son realmente positivos.
recall_score(y_test, y_pred) #Cuantos valores positivos son correctamente clasificados.
f1_score(y_test, y_pred) #

cr = classification_report(y_test, y_pred)
print(cr)
  
#Curva Roc
fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.title('Característica operativa del receptor.')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Curva ROC para kNN')
plt.show()

#Resultados.
data_output = pd.concat([pd.DataFrame(x_test.index),
                         pd.DataFrame(y_test).reset_index(),
                         pd.DataFrame(y_pred)], axis = 1)

data_output = data_output.iloc[:,[1,2,3]]
data_output.columns = ["anonID","success_real","success_pred"]
data_output

#Knn : No dio los mejores resultados para está clasificación.
