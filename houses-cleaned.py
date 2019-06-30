# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 20:14:01 2019
Kaggle score: 0.120
@author: Priyam
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge,ElasticNet
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

    #
    #Add some features
    df['Age'] = 2015 - df['YearRemodAdd']
    df['OriginalAge'] = 2015 - df['YearBuilt']
    df['BathsPrRms'] = (df['BsmtFullBath']+df['BsmtHalfBath']+df['FullBath']+df['HalfBath'])/df['TotRmsAbvGrd']
    df['RoomSize'] = df['GrLivArea']/df['TotRmsAbvGrd']
#    
#    
    #Get Dummies for categorical variables
    categorical = pd.get_dummies(df[categoricals].astype(str))
    numerical = df.drop(categoricals,1).astype(float)
    numerical.fillna(0,inplace=True)
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
rs = RobustScaler()
rs.fit_transform(processed_all)
test = processed_all[train_length:]
train = processed_all[:train_length]

#Prepare training data
df.SalePrice = np.log1p(df.SalePrice)
y = df.SalePrice.astype(float)
X = train


#Split training set into training and validation sets
xtr,xval,ytr,yval = train_test_split(X,y)

ridge = Ridge(alpha=100)
ridge.fit(xtr,ytr)
ridge_val = ridge.score(xval,yval)

elastic = ElasticNet(alpha=0.0025,max_iter=10000)
elastic.fit(xtr,ytr)
elastic_val = elastic.score(xval,yval)

gbr = GradientBoostingRegressor(learning_rate=0.05,n_estimators=256)
gbr.fit(xtr,ytr)
gbr_val = gbr.score(xval,yval)


##So far so good. Let's optimize.
#pset = [1024,2048,4096]
#pscores = []
#for p in pset:
#    rftest = GradientBoostingRegressor(learning_rate=0.05,n_estimators=p)
#    rftest.fit(xtr,ytr)
#    score_train = rftest.score(xtr,ytr)
#    score_test = rftest.score(xval,yval)
#    pscores.append([p,score_train,score_test])
    
#So far so good. Let's optimize.
#pset = np.arange(10,30,3)*0.0001
#pscores = []
#for p in pset:
#    lftest = ElasticNet(alpha = p,max_iter=10000)
#    lftest.fit(xtr,ytr)
#    score_train = lftest.score(xtr,ytr)
#    score_test = lftest.score(xval,yval)
#    pscores.append([p,score_train,score_test])


#Retrain optimized model on full training set
model = GradientBoostingRegressor(learning_rate=0.05,n_estimators=2048,loss='huber')
model.fit(X,y)
score_training = model.score(X,y)

model1 = ElasticNet(alpha=0.0025,max_iter=10000)
model1.fit(X,y)
score1_training = model1.score(X,y)

model2 = Ridge(alpha=100)
model2.fit(X,y)
score2_training = model2.score(X,y)


#Prepare test data
X_test = test
test_pred = model.predict(X_test)

test_pred = (model1.predict(X_test)+model2.predict(X_test)+model.predict(X_test))/3.0

predicted_prices = np.exp(test_pred)-1
answers = pd.DataFrame({'Id':X_test.index+1461,'SalePrice':predicted_prices})
s = answers.to_csv('house-prices-advanced-regression-techniques/housing-answers-629.csv',index=False,header=True)
