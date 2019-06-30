# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 12:14:01 2019
Kaggle score: 0.132
@author: Priyam
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor


categoricals = ['MSSubClass','MSZoning','Street','Alley','LotShape','LandContour',
                'Utilities','LotConfig','LandSlope','Neighborhood',
                'Condition1','Condition2','BldgType','HouseStyle',
                'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
                'ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond',
                'BsmtExposure','BsmtFinType1','BsmtFinType2','Heating','HeatingQC',
                'CentralAir','Electrical','KitchenQual','Functional','FireplaceQu',
                'GarageType','GarageFinish','GarageQual','GarageCond','PavedDrive',
                'PoolQC','Fence','MiscFeature','SaleType','SaleCondition',
                'MiscVal','MoSold'
                ]



def processREdata(df):
    #Make sure to separate SalePrice from training data before using this
    df['LotArea'] = np.log1p(df['LotArea'])
    #Impute to fill missing lot frontage
    lot_imp = IterativeImputer()
    lots = df[['LotFrontage','LotArea']]
    lot_imp.fit(lots)
    lots_calc = lot_imp.transform(lots)
    df[['LotFrontage','LotArea']]=lots_calc  
    #Impute to fill missing garage year built
    history = df[['YearBuilt','YearRemodAdd','GarageYrBlt']]
    yr_imp = IterativeImputer()
    yr_imp.fit(history)
    df[['YearBuilt','YearRemodAdd','GarageYrBlt']] = yr_imp.transform(history)
    #Replace missing masonry area with zero
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)

    #Add some features
    df['Age'] = 2015 - df['YearRemodAdd']
    df['OriginalAge'] = 2015 - df['YearBuilt']
    
    #Get Dummies for categorical variables
    categorical = pd.get_dummies(df[categoricals].astype(str))
    numerical = df.drop(categoricals,1)
    dg = pd.concat([numerical,categorical],1)
    return dg


#Now load data
df = pd.read_csv('house-prices-advanced-regression-techniques/train.csv')
test_data = pd.read_csv('house-prices-advanced-regression-techniques/test.csv')

#Concatenate train and test data to process together, then split back apart
train_length = len(df)
train_data = df.drop('SalePrice',1)
all_data = pd.concat([train_data,test_data],0)
processed_all = processREdata(all_data)
test = processed_all[train_length:]
train = processed_all[:train_length]

#Prepare training data
df.SalePrice = np.log1p(df.SalePrice)
y = df.SalePrice.astype(float)
X = train

#Split training set into training and validation sets
xtr,xval,ytr,yval = train_test_split(X,y)


gbr = GradientBoostingRegressor(learning_rate=0.1,n_estimators=100)
gbr.fit(xtr,ytr)
score_tr = gbr.score(xtr,ytr)
score_te = gbr.score(xval,yval)

#
##So far so good. Let's optimize.
#pset = np.arange(30,75,5)*0.001
#pscores = []
#for p in pset:
#    rftest = GradientBoostingRegressor(learning_rate=0.055,n_estimators=256)
#    rftest.fit(xtr,ytr)
#    score_train = rftest.score(xtr,ytr)
#    score_test = rftest.score(xval,yval)
#    pscores.append([p,score_train,score_test])
#    

#Retrain optimized model on full training set
model = GradientBoostingRegressor(learning_rate=0.055,n_estimators=256)
model.fit(X,y)
score_training = model.score(X,y)

#Prepare test data. There are still 10 missing values (garage or basement)
# should be ok to fill with zero
X_test = test
X_test = X_test.fillna(0)

test_pred = model.predict(X_test)
predicted_prices = np.exp(test_pred)-1
answers = pd.DataFrame({'Id':X_test.index+1461,'SalePrice':predicted_prices})
s = answers.to_csv('house-prices-advanced-regression-techniques/housing-answers-629.csv',index=False,header=True)