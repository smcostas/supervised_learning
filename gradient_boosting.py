# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:30:17 2024

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

## voy a intentar balanceando haciendo un over sample

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Estandarización
    ('gb', GradientBoostingClassifier(random_state=42))  # voy a usar gradient boost
])


param_grid = {
    'gb__loss': ['log_loss','exponential'],
    'gb__learning_rate': [0.01,0.1],
    'gb__max_depth': [5,10,15],
    'gb__min_samples_split': [2, 5,10],
    'gb__min_samples_leaf': [1, 2, 4],
    'gb__max_features': [None,'sqrt', 'log2'],
    'gb__subsample': [0.6, 0.8, 1.0]
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
df_top10.columns
pd.set_option('display.max_columns', None)

print(df_top10)
params_dict = df_top10.params[74]


pipeline.set_params(**params_dict)

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_train) ## 
y_pred_test= pipeline.predict(X_test) ## en el testeo


print(classification_report(y_train, y_pred))
print(classification_report(y_test,y_pred_test))


print(accuracy_score(y_test,y_pred_test))
y_pred_prob_train = pipeline.predict_proba(X_train)[:, 1]
y_pred_prob_test = pipeline.predict_proba(X_test)[:, 1]

auc_score = roc_auc_score(y_train, y_pred_prob_train)
print(f"AUC del modelo seleccionado: {auc_score}")
fpr, tpr, thresholds = roc_curve(y_train, y_pred_prob_train)
J = tpr - fpr ## maximizo este valor
optimal_threshold_index = J.argmax()
optimal_threshold = thresholds[optimal_threshold_index]
optimal_threshold

## no me gusta..

y_pred_optimal = (y_pred_prob_train >= optimal_threshold).astype(int)
y_pred_optimal_test = (y_pred_prob_test >= optimal_threshold).astype(int)
print(classification_report(y_train,y_pred_optimal))
print(classification_report(y_test,y_pred_optimal_test))


from sklearn.utils import resample

def undersample(X, y, traget_name):
    # Combinar X e y
    data = pd.concat([X, y], axis=1)
    majority_class = data[data[traget_name] == 0]
    minority_class = data[data[traget_name] == 1]
    majority_downsampled = resample(majority_class, replace=False, n_samples=len(minority_class), random_state=42)
    balanced_data = pd.concat([majority_downsampled, minority_class])
    return balanced_data.drop(traget_name, axis=1), balanced_data[traget_name]


X_train_under, y_train_under = undersample(X_train, y_train, 'diabetes')


model = GradientBoostingClassifier(random_state=42) ## voy a probar sin estandarizar a ver que ocurre
param_grid = {
    'loss': ['log_loss','exponential'],
    'min_samples_split': [2, 5,10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None,'sqrt'],
    'subsample': [0.6, 0.8, 1.0],
    'n_estimators':[1000]
}


grid_search = RandomizedSearchCV(
    estimator = model,
    param_distributions = param_grid,
    scoring = 'neg_log_loss',
    n_jobs = -1,
    n_iter = 100,
    cv=5
)

grid_search.fit(X_train_under, y_train_under)

results = grid_search.cv_results_
df = pd.DataFrame(results)
#df.columns
df_sorted = df.sort_values(by='rank_test_score', ascending=True)
df_top10 = df_sorted[df_sorted['rank_test_score'] <= 10]
df_top10.columns


print(df_top10)

params = df_top10.params[19]

model.set_params(**params)
model.fit(X_train_under, y_train_under)
pred_under = model.predict(X_train_under)
pred_test = model.predict(X_test)

print(classification_report(y_train_under, pred_under))
print(classification_report(y_test, pred_test))

## balancearla de esta forma no sirve

sum(y)/len(y)

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


## voy probar elegir mejores parametros a ver si lo puedo mejorar




pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Estandarización
    ('xgbc', XGBClassifier(n_jobs = -1, scale_pos_weight=1/.6))   ## con este peso logro un buen modelo . voy a hacer seleccion con el xgbc
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
    scoring = 'neg_log_loss',
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
df_top10

best_params = df_top10.params[25]

pipeline.set_params(**best_params)

pipeline.fit(X_train,y_train)

print(pipeline)
pred = pipeline.predict(X_train)
pred_test = pipeline.predict(X_test)


print(classification_report(y_train, pred))
print(classification_report(y_test, pred_test))

accuracy_score(y_test, pred_test)

## entreno con toda la info para luego predecir 

pipeline.fit(X,y) ### entrenamiento final con todos los datos
preds_final = pipeline.predict(X) ### para confirmar que todo va bien

print(classification_report(y, preds_final)) ## no cambia nada 

df_test = pd.read_csv('diabetes_prediction_dataset_test.csv')



if df_test['smoking_history'].dtype == 'object':
    smoking_history_dict = {'No Info': 0, 'never': 1, 'ever': 2, 'former': 3, 'not current': 4, 'current': 5 }
    df_test['smoking_history'] = df_test['smoking_history'].map(smoking_history_dict)


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

pd.DataFrame.to_csv(entregable,'predichos1.csv', index=False)






## voy a repetir usando SMOTE
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import make_pipeline

over = SMOTE(sampling_strategy=0.1)
under = RandomUnderSampler(sampling_strategy=0.5)
steps = [('o', over), ('u', under)]
pipe_bal = make_pipeline(over,under)

# Aplica el pipeline para balancear los datos
X_bal, y_bal = pipe_bal.fit_resample(X_train, y_train)
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Estandarización
    ('xgbc', XGBClassifier(n_jobs = -1))   ## con este peso logro un buen modelo . voy a hacer seleccion con el xgbc
])

##pipeline.set_params(**best_params)

pipeline.fit(X_bal, y_bal)

pred_bal = pipeline.predict(X_bal)
pred_final = pipeline.predict(X_test)

print(classification_report(y_bal, pred_bal))
print(classification_report(y_test, pred_final))


# otra forma de intentar balancerlo
from imblearn.ensemble import BalancedBaggingClassifier

model = BalancedBaggingClassifier(base_estimator=GradientBoostingClassifier(), n_estimators=1000)


model.fit(X_train, y_train)

y_pred = model.predict(X_train)
y_pred_test = model.predict(X_test)

print(classification_report(y_train, y_pred))
print(classification_report(y_test, y_pred_test))

