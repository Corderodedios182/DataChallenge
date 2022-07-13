# -*- coding: utf-8 -*-
"""
Created on Wed Jul 13 13:55:26 2022

@author: cflorelu
"""

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
