# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 12:49:35 2024

@author: santi
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
os.chdir('Downloads/ap supervisado')
#from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
train_df = pd.read_csv('diabetes_prediction_dataset_train-labeled.csv')

y = train_df['diabetes']
X = train_df.iloc[:,:-1]

X = X.drop('patient',axis = 1)

## one hot
types = X.dtypes
cols_cat  = types[types == 'object'].index.to_list()
cols_num  = types[types != 'object'].index.to_list()
X[cols_cat].info() ## solo tengo smoking y gender

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_cat = encoder.fit_transform(X[cols_cat])
new_columns = []
for col, col_values in zip(cols_cat, encoder.categories_):
  for col_value in col_values:
    new_columns.append('{}={}'.format(col, col_value))

X = np.hstack([X_cat, X[cols_num].values])
new_columns.extend(cols_num)

X =  pd.DataFrame(data=X, columns=new_columns)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


pca = PCA(n_components=11) 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_train_pca = pca.fit_transform(X_train_scaled)

varianza_explicada = pca.explained_variance_ratio_*100
sum(varianza_explicada) ## 11 ejes explican el 90.55 


model = XGBClassifier(n_jobs = -1,random_state = 42)
model.fit(X_train_pca, y_train)
X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)
preds = model.predict(X_test_pca)

print(classification_report(y_test, preds))
accuracy_score(y_test, preds)
    