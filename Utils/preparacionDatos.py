# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import LabelEncoder

def dataClean(data):
    
    """
    extracción, formato columnas, limpieza columnas
    """

    def edad(birthdate):
        return int(((datetime.now() - birthdate).days)/365.2425)

    data_columns = data.columns.str.split(';',expand=True)
    
    data = data.iloc[:,0].str.split(';',expand=True)
    
    data.columns = ['anonID','birthdate','routeDate','region','gender','areaWealthLevel','areaPopulation',
                    'badWeather','weatherRestrictions','routeTotalDistance','numberOfShops','marketShare',
                    'avgAreaBenefits','timeFromAvg','advertising','employeeLYScore','employeeTenure','employeePrevComps','success']
    
    data = data.reset_index().drop(["index"], axis =1)
        
    #Formato de columnas#
    
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
    
    #filters = [(data.edad >= 30) & (data.edad <= 39),
    #           (data.edad >= 40) & (data.edad <= 49),
    #           (data.edad >= 50) & (data.edad <= 99),]
    
    #values = [30, 40, 50]
    
    #data["edad"] = np.select(filters, values)
    
    filters_marketShare = [(data.marketShare >= 10) & (data.marketShare <= 19),
                           (data.marketShare >= 20) & (data.marketShare <= 29),
                           (data.marketShare >= 30) & (data.marketShare <= 39),
                           (data.marketShare >= 40) & (data.marketShare <= 49),
                           (data.marketShare >= 50) & (data.marketShare <= 59),
                           (data.marketShare >= 60) & (data.marketShare <= 69),
                           (data.marketShare >= 70) & (data.marketShare <= 79),
                           (data.marketShare >= 80) & (data.marketShare <= 89),
                           (data.marketShare >= 90) & (data.marketShare <= 101)]
    
    values = [10,20,30,40,50,60,70,80,90]
    
    data["range_marketShare"] = np.select(filters_marketShare, values)
    
    filters_avgAreaBenefits = [(data.avgAreaBenefits > 0) & (data.avgAreaBenefits < 20),
                               (data.avgAreaBenefits >= 20) & (data.avgAreaBenefits < 31),
                               (data.avgAreaBenefits > 30) & (data.avgAreaBenefits < 41),
                               (data.avgAreaBenefits > 40) & (data.avgAreaBenefits < 1000)]
    
    values = [10,20,30,40]
    
    data["range_avgAreaBenefits"] = np.select(filters_avgAreaBenefits, values)
        
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
                       "range_marketShare","range_avgAreaBenefits",
                       "avgAreaBenefits","timeFromAvg","advertising","employeeLYScore","employeeTenure","employeePrevComps","success"]]
    
    return data
    
########################
#Datos para exploración#
########################


#Variables categoricas, respecto a la distribución (0,1)
def grupos(data_filter ,grupo = ['region','success'], num_grupos = 2):
    if num_grupos == 1:
        return data_filter.groupby(grupo).count().iloc[:,:1]
    else :
        return data_filter.groupby(grupo).agg({'success': 'count'}).groupby(level=0).apply(lambda x: round(100 * x / float(x.sum()),1)) 

####################
#Datos para Modelos#
####################
def dataModel(data_filter):
    
    """ 
    preprocesamiento de datos para modelo de machine learning, selección de columnas e ingeniería de características.
    """
    
    data_model = data_filter
    
    data_model.index = data_filter["anonID"]
    
    #data_model = data_filter.loc[:,["edad","region","areaWealthLevel",
    #                                 "badWeather",# "weatherRestrictions",# "areaPopulation", 
    #                                 "routeTotalDistance","numberOfShops","marketShare","avgAreaBenefits",
    #                                 "timeFromAvg","advertising","employeeLYScore","employeeTenure","employeePrevComps","success"]]
    
    data_model = data_filter.loc[:,["avgAreaBenefits","routeTotalDistance","advertising","employeeLYScore",
                                    "range_marketShare","range_avgAreaBenefits","timeFromAvg","employeePrevComps","numberOfShops",
                                    "success"]]

    #le = LabelEncoder()
    #data_model["region"] = le.fit_transform(data_model["region"])
    #data_model["gender"] = le.fit_transform(data_model["gender"])
    #data_model["areaWealthLevel"] = le.fit_transform(data_model["areaWealthLevel"])
    #data_model["badWeather"] = le.fit_transform(data_model["badWeather"])
    #data_model["weatherRestrictions"] = le.fit_transform(data_model["weatherRestrictions"])
    
    return data_model
