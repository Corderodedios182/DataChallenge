# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:27:09 2022

@author: cflorelu
"""

import pandas as pd
import json
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
import warnings
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
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, roc_curve
import shap

from Utils import preparacionDatos

with open('Inputs/deliveryDatasetChallenge.json') as json_file:
    data = json.load(json_file)
    
    print("Type:", type(data))

data = pd.DataFrame(data)

data = preparacionDatos.dataClean(data)

#Datos preproceso ML
data_model = preparacionDatos.dataModel(data)

preparacionDatos.grupos(data,['success'],num_grupos=1) #Lo valores -1 son los que queremos predecir.

#Datos de prueba
data_prueba = data_model[data_model["success"] == -1]

#Datos de test y datos de entrenamiento.
data_model = data_model[data_model["success"] != -1]

aux = data_model

#Datos estandarizados
scaler = StandardScaler()
scaler.fit(aux.iloc[:,:-1])
x_scaled, y = scaler.transform(aux.iloc[:,:-1]), aux.success

x_train_sd, x_test_sd, y_train, y_test = train_test_split(x_scaled, y, test_size = 0.2, random_state = 123, stratify = y)
y_test.value_counts(normalize = True)
y_train.value_counts(normalize = True)

#Datos Entrenamiento y Test sin estandarizar
x, y = aux.iloc[:,:-1], aux.success
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 123, stratify = y)
y_test.value_counts(normalize = True)
y_train.value_counts(normalize = True)
x_train.head()

#Mejor modelo Xgboost.
modelo_xgb = xgb.XGBClassifier()

modelo_xgb.fit(x_train, y_train)
y_pred_xgb = modelo_xgb.predict(x_test)

accuracy = float(np.sum(y_pred_xgb == y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))

confusion_matrix(y_test, y_pred_xgb)
y_test.value_counts()

#Curva Roc
fpr, tpr, threshold = roc_curve(y_test, y_pred_xgb)
roc_auc = auc(fpr, tpr)
roc_auc

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

#metrics 
explainer = shap.TreeExplainer(modelo_xgb)
shap_values = explainer.shap_values(x_train)
shap.summary_plot(shap_values, x_train)

#Tuning hyperparametros XGBoost
modelo_xgbr = xgb.XGBClassifier()

grid_values_xgbr = {'objective':['binary:logistic'],
                    'learning_rate': [0.1, 0.01, 0.05], #so called `eta` value
                    'max_depth': range (2, 10, 1),
                    'min_child_weight': [11],
                    'silent': [1],
                    'subsample': [0.7,0.8,0.9],
                    'colsample_bytree': [0.4, 0.5, 0.7],
                    'n_estimators': range(60, 220, 40), #number of trees, change it to 1000 for better results
                    'missing':[-999],
                    'seed': [42]}

grid_xgbr_acc = GridSearchCV(modelo_xgbr,
                             param_grid = grid_values_xgbr,
                             n_jobs=5, 
                             cv=10,
                             scoring = 'roc_auc',
                             verbose=2, 
                             refit=True)

grid_xgbr_acc.fit(x_train_sd, y_train)
      
params_xgbr = grid_xgbr_acc.best_params_

#Mejo modelo Lightgbm
modelo_lgb = lgb.LGBMClassifier(objective="binary")

modelo_lgb.fit(x_train,y_train)
y_predict_lgb = modelo_lgb.predict(x_test)

accuracy = float(np.sum(y_predict_lgb == y_test))/y_test.shape[0]
print("accuracy: %f" % (accuracy))

confusion_matrix(y_test, y_predict_lgb)
y_test.value_counts()

#Curva Roc
fpr, tpr, threshold = roc_curve(y_test, y_predict_lgb)
roc_auc = auc(fpr, tpr)
roc_auc

#metrics 
explainer = shap.TreeExplainer(modelo_lgb)
shap_values = explainer.shap_values(x_train)
shap.summary_plot(shap_values, x_train)

#tree visualization
graph = lgb.create_tree_digraph(modelo_lgb)
graph.graph_attr.update(size="110,110")
graph

#Tuning hyperparametros LGBMClassifier
rs_params = {
            'learning_rate': (0.01, 1.0),
            'bagging_fraction': (0.5, 0.8),
            'bagging_frequency': (5, 8),
            'feature_fraction': (0.5, 0.8),
            'max_depth': (10, 13),
            'min_data_in_leaf': (90, 120),
            'num_leaves': (800,1200)
            }

rs_cv = RandomizedSearchCV(estimator=lgb.LGBMClassifier(), param_distributions=rs_params, cv = 5, n_iter=100,verbose=1)
rs_cv.fit(x_train_sd, y_train)
      
params_rs_cv = rs_cv.best_params_

#Modelo KNN
#---Grafica variando el numero de n_neighbors y su presición.
n_neighbors = range(1,9)
train  = []
test = []

for i in n_neighbors:
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    
    train.append(knn.score(x_train, y_train))
    test.append(knn.score(x_test, y_test))

plt.rcParams['figure.figsize'] = [8, 5]
line_up, = plt.plot(n_neighbors, train , label='Entrenamiento Accuracy')
line_down, = plt.plot(n_neighbors, test, label='Testeo Accuracy')
plt.title("KNN : Variación de numero de cluster.")
plt.legend([line_up, line_down], ['Entrenamiento Accuracy', 'Testeo Accuracy'])

plt.annotate('Overfiting: Bajo sesgo, alta varianza.', xy = (2, .982), arrowprops = {'facecolor':'gray', 'width': 3, 'shrink': 0.03})
plt.annotate('Optimo', xy = (4, .96), arrowprops = {'facecolor':'gray', 'width': 3, 'shrink': 0.03})
plt.annotate('Underfiting : Alto sesgo, baja varianza.', xy = (2, .95), arrowprops = {'facecolor':'gray', 'width': 3, 'shrink': 0.03})

#---Para varios valores de k en kNN, examinemos también cómo van a ser nuestras pérdidas, sesgos y varianzas
bias_KnnClass, var_KnnClass,error_KnnClass, = [], [], []
for k in range(1,8):
    clf_knn = KNeighborsClassifier(n_neighbors=k)
    avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(clf_knn, x_train.values, y_train.values, x_test.values, y_test.values, loss = '0-1_loss', random_seed = 123)
    bias_KnnClass.append(avg_bias)
    var_KnnClass.append(avg_var)
    error_KnnClass.append(avg_expected_loss)
    print(f"n_neighbors {k}")
    print(f"Pérdida media esperada {avg_expected_loss}")
    print(f"Media bias {avg_bias}")
    print(f"Media variance {avg_var}")
    print("------")

plt.rcParams['figure.figsize'] = [16, 5]
plt.plot(range(1,8), error_KnnClass, 'red', label = 'total_error',linestyle='dashed')
plt.plot(range(1,8), bias_KnnClass, 'brown', label = 'bias^2')
plt.plot(range(1,8), var_KnnClass, 'yellow', label = 'varianza')
plt.xlabel('Complejidad del modelo con (K)')
plt.ylabel('Error')
plt.legend()

#----Mejor modelo analizando knn con Sesgo y Varianza.
clf_knn = KNeighborsClassifier(n_neighbors=4)
clf_knn.fit(x_train, y_train)
score = clf_knn.score(x_train, y_train)
print("Score: ", score)

y_pred = clf_knn.predict(x_test)

avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(clf_knn, x_train.values, y_train.values, x_test.values, y_test.values, loss = '0-1_loss', random_seed = 123)
print('Pérdida media esperada: %.3f' % avg_expected_loss)
print('Media Bias: %.3f' % avg_bias)
print('Media Variance: %.3f' % avg_var)
print('Skelarn 0-1 perdida : %.3f' % zero_one_loss(y_test, y_pred))

#----Métricas de Clásificación para nuestro modelo knn = 4.
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

#----Curva Roc
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
