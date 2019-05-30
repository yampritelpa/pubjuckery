# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:14:01 2019

@author: Priyam
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error





df = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')

df.SalePrice = np.log1p(df.SalePrice)
df.LotArea = np.log1p(df.LotArea)
print("before feature extraction: ")
plt.matshow(df.corr());plt.show()
#MAPS

def buildhoodmap(df):
    nbm = df[['Neighborhood','SalePrice']].sort_values('SalePrice',ascending=False)
    #
    nbm = pd.pivot_table(nbm,values='SalePrice',columns='Neighborhood',
                         aggfunc='mean').sort_values(ascending=True)
    nbm = nbm.reset_index('SalePrice')
    hood_sorted = nbm.Neighborhood
    
    hoodnum = np.array(hood_sorted.index)
    hoodmap = {}
    for k in range(len(hoodnum)):
        hoodmap.update({hood_sorted[k]:hoodnum[k]})
    return(hoodmap)

def maps(df,hoodmap):
    Qnummap = {'Ex':4,'Gd':3,'TA':2,'Fa':1,'Po':0}
    df['ExQnum']=df['ExterQual'].map(Qnummap)
    df['hood']=df['Neighborhood'].map(hoodmap)
    return(df)

###
hoodmap = buildhoodmap(df)
df = maps(df,hoodmap)

correlations = df.corr().SalePrice
indicators = [n for n in range(1,len(correlations)) if np.abs(correlations[n])>0.53]
indmat = df.iloc[:,indicators]
indicators = list(indmat.columns)

numinds = ['LotArea','MasVnrArea','OverallQual','YearBuilt','YearRemodAdd',
'TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd',
'GarageCars','GarageArea','GarageYrBlt','ExQnum','hood']

num_ind = df[numinds]
num_ind = num_ind.fillna(0)                    
plt.matshow(num_ind.corr());plt.show()                      

x_train = num_ind
y_train = df.SalePrice.astype(float)

x_tr, x_te, y_tr, y_te = train_test_split(x_train,y_train,test_size=0.75,random_state=69)


###ML ALGORITHMS
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet

###Function to cycle through linear algorithms and calculate error from train/test split

models = []
modelscores = []
models.append(('linreg',LinearRegression()))
models.append(('lasso',Lasso()))
models.append(('ridge',Ridge()))
models.append(('elastic',ElasticNet()))
models.append(('forest',RandomForestRegressor()))
models.append(('gbr',GradientBoostingRegressor()))

for name, model in models:
    model.fit(x_tr,y_tr)
    y_pp = model.predict(x_te)
    s = np.sqrt(mean_squared_error(y_te,y_pp))
    modelscores.append([name,s])

###Of these models, GradientBoost outperforms so I will optimize further via regularization
pset = [5e-3,1e-2,2e-2]
pscores = []
for p in pset:
    rftest = GradientBoostingRegressor(learning_rate=1e-2,n_estimators=1000)
    rftest.fit(x_tr,y_tr)
    y_p = rftest.predict(x_te)
    s = np.sqrt(mean_squared_error(y_te,y_p))
    pscores.append([p, s])
    
###From this assessment, I conclude that lr=0.01 and n_est=1000 are appropriately tuned hyperparameters

final_model = GradientBoostingRegressor(learning_rate=0.025,n_estimators=500)
final_model.fit(x_train,y_train)    
    
###Moving onto test data



test_data = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')
test_data = maps(test_data,hoodmap)

#test_data = test_data.dropna(0)
X_test = test_data[numinds]
X_test = X_test.fillna(0)
test_pred = (final_model.predict(X_test))


answers = pd.DataFrame({'Id':X_test.index+1461,'SalePrice':test_pred})
s = answers.to_csv('house-prices-advanced-regression-techniques/housing-answers.csv',index=False,header=True)