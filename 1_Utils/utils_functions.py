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

def edad(birthdate):
    return int(((datetime.now() - birthdate).days)/365.2425)

with open('deliveryDatasetChallenge.json') as json_file:
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
data["month"] = data["routeDate"].dt.month
data["day"] = data["routeDate"].dt.day
data["year"] = data["routeDate"].dt.year

#Reacomodo
data = data.loc[:,["anonID","birthdate","edad","routeDate","month","day","year","region","gender","areaWealthLevel","badWeather","weatherRestrictions",
                   "region-gender-areaWealthLevel-badWeather-weatherRestrictions","areaPopulation","routeTotalDistance","numberOfShops","marketShare",
                   "avgAreaBenefits","timeFromAvg","advertising","employeeLYScore","employeeTenure","employeePrevComps","success"]]

######################
#Exploración de datos#
######################




