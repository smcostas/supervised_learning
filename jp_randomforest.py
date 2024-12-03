# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 11:29:59 2024

@author: santi
"""

import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import seaborn as sns


#os.chdir('Downloads/ap supervisado')
train_df = pd.read_csv('diabetes_prediction_dataset_train-labeled.csv')
train_df.columns
sns.boxplot(x = 'blood_glucose_level', data =train_df )

train_df = train_df[train_df['blood_glucose_level'] < 250]
y = train_df['diabetes']
X = train_df.iloc[:,:-1]

X = X.drop('patient',axis = 1)




types = X.dtypes
cols_cat  = types[types == 'object'].index.to_list()
cols_num  = types[types != 'object'].index.to_list()
X[cols_cat].info() ## solo tengo smoking y gender
from sklearn.preprocessing import OneHotEncoder
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
X_train.columns


pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Estandarización
    ('rf', RandomForestClassifier(random_state=42))  # voy a usar gradient boost
])

Best_Parameters = {'rf__max_depth': 20, 
                   'rf__min_samples_leaf': 2, 
                   'rf__min_samples_split': 2, 
                   'rf__n_estimators': 100
                   }
pipeline.set_params(**Best_Parameters)


pipeline.fit(X_train, y_train)

preds_train = pipeline.predict(X_train)
preds_test = pipeline.predict(X_test)

print(classification_report(y_train, preds_train))
print(classification_report(y_test, preds_test))
accuracy_score(y_test, preds_test)

### reentreno con todos los datos para tener toda la info !! 
pipeline.fit(X, y)
preds = pipeline.predict(X)
print(classification_report(y, preds))
accuracy_score(y, preds)



df_test = pd.read_csv('diabetes_prediction_dataset_test.csv')

X = df_test.iloc[:,:-1]

X = X.drop('patient',axis = 1)

types = X.dtypes
cols_cat  = types[types == 'object'].index.to_list()
cols_num  = types[types != 'object'].index.to_list()
X[cols_cat].info() ## solo tengo smoking y gender
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_cat = encoder.fit_transform(X[cols_cat])
new_columns = []
for col, col_values in zip(cols_cat, encoder.categories_):
  for col_value in col_values:
    new_columns.append('{}={}'.format(col, col_value))

X = np.hstack([X_cat, X[cols_num].values])
new_columns.extend(cols_num)

X =  pd.DataFrame(data=X, columns=new_columns)


diabetes_final = pipeline.predict(X)

df_test['diabetes'] = diabetes_final

entregable = df_test.loc[:,['patient','diabetes']]

pd.DataFrame.to_csv(entregable,'predichos8.csv', index=False)

