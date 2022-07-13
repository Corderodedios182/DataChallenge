# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 13:55:26 2022

@author: cflorelu
"""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from mlxtend.evaluate import bias_variance_decomp
from sklearn.metrics import zero_one_loss

def mlClfKnn(data_model):

    x, y = data_model.iloc[:,:-1], data_model.success
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 123, stratify = y)
    
    y_test.value_counts(normalize = True)
    y_train.value_counts(normalize = True)
    
    clf_knn = KNeighborsClassifier(n_neighbors=4)
    clf_knn.fit(x_train, y_train)
    score = clf_knn.score(x_train, y_train)
    print("Score: ", score)
    
    y_pred = clf_knn.predict(x_test)
    
    #avg_expected_loss, avg_bias, avg_var = bias_variance_decomp(clf_knn, x_train.values, y_train.values, x_test.values, y_test.values, loss = '0-1_loss', random_seed = 123)
    #print('Pérdida media esperada: %.3f' % avg_expected_loss)
    #print('Media Bias: %.3f' % avg_bias)
    #print('Media Variance: %.3f' % avg_var)
    #print('Skelarn 0-1 perdida : %.3f' % zero_one_loss(y_test, y_pred))
    
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
    
    data_output = pd.concat([pd.DataFrame(x_test.index),
                             pd.DataFrame(y_test).reset_index(),
                             pd.DataFrame(y_pred)], axis = 1)

    data_output = data_output.iloc[:,[1,2,3]]
    data_output.columns = ["anonID","success_real","success_pred"]
    data_output

    return data_output
