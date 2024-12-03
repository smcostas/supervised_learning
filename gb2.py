# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 19:23:51 2024

@author: santi
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
#os.chdir('Downloads/ap supervisado')
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score



train_df = pd.read_csv('diabetes_prediction_dataset_train-labeled.csv')



if train_df['smoking_history'].dtype == 'object':
    smoking_history_dict = {'No Info': 0, 'never': 1, 'ever': 2, 'former': 3, 'not current': 4, 'current': 5 }
    train_df['smoking_history'] = train_df['smoking_history'].map(smoking_history_dict)


y = train_df['diabetes']
X = train_df.iloc[:,:-1]

X = X.drop('patient',axis = 1)
X = X.loc[:,['age', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from xgboost import XGBClassifier

pipeline = pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Estandarización
    ('xgbc', XGBClassifier(scale_pos_weight=1/.6, n_jobs = -1))   ## con este peso logro un buen modelo . voy a hacer seleccion con el xgbc
])


pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)

print(classification_report(y_train, y_pred))
print(classification_report(y_test, y_pred_test))

pipeline.fit(X, y)
y_pred_test = pipeline.predict(X)
print(classification_report(y, y_pred_test))


df_test = pd.read_csv('diabetes_prediction_dataset_test.csv')

y = df_test['diabetes']
X = df_test.iloc[:,:-1]


if df_test['smoking_history'].dtype == 'object':
    smoking_history_dict = {'No Info': 0, 'never': 1, 'ever': 2, 'former': 3, 'not current': 4, 'current': 5 }
    df_test['smoking_history'] = df_test['smoking_history'].map(smoking_history_dict)


X = df_test.iloc[:,:-1]

X = X.drop('patient',axis = 1)

X = X.loc[:,['age', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']]

diabetes_final = pipeline.predict(X)
sum(diabetes_final)


df_test['diabetes'] = diabetes_final

entregable = df_test.loc[:,['patient','diabetes']]

pd.DataFrame.to_csv(entregable,'predichos3.csv', index=False)

### falta seleccion
## ahi va 

from sklearn.metrics import f1_score, make_scorer
f1_minority_scorer = make_scorer(f1_score, pos_label=1)

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Estandarización
    ('xgbc', XGBClassifier(n_jobs = -1,n_estimators = 1000))   ## con este peso logro un buen modelo . voy a hacer seleccion con el xgbc
])

param_grid = {
    'xgbc__eta': [1, 0.1],
    'xgbc__max_depth': [5,10,None],
    'xgbc__gamma' :[0,0.1,0.5],
    'xgbc__subsample':[0.5,1],
    'xgbc__colsample_bytree':[0.5,1],
    'xgbc__reg_lambda':[0, 0.1 , 1,100],
    'xgbc__reg_alpha':[ 0, 0.1, 1, 100],
}


grid_search = RandomizedSearchCV(
    estimator = pipeline,
    param_distributions = param_grid,
    scoring = 'accuracy',
    n_jobs = -1,
    n_iter = 100,
    cv=5
)

grid_search.fit(X_train, y_train)
results = grid_search.cv_results_
df = pd.DataFrame(results)
#df.columns
df_sorted = df.sort_values(by='rank_test_score', ascending=True)
df_top10 = df_sorted[df_sorted['rank_test_score'] <= 10]

params = df_top10.params[58]
pipeline.set_params(**params)

pipeline.fit(X_train, y_train)
preds_train = pipeline.predict(X_train)
preds_test = pipeline.predict(X_test)

print(classification_report(y_train, preds_train))
print(classification_report(y_test, preds_test))

pipeline.fit(X, y)
preds_f = pipeline.predict(X)

print(classification_report(y, preds_f))
accuracy_score(y, preds_f)


