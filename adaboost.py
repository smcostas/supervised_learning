# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:09:09 2024

@author: santi
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.svm import SVC
os.chdir('Downloads/ap supervisado')
from sklearn.pipeline import Pipeline
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
tree1 = DecisionTreeClassifier(max_depth = 1, criterion = 'gini', splitter = 'best')
tree2 = DecisionTreeClassifier(max_depth = 1, criterion = 'log_loss', splitter = 'best')
tree3 = DecisionTreeClassifier(max_depth = 1, criterion = 'log_loss', splitter = 'random')
tree4 = DecisionTreeClassifier(max_depth = 3, criterion = 'gini', splitter = 'best')
tree5 = DecisionTreeClassifier(max_depth = 3, criterion = 'log_loss', splitter = 'best')
tree6 = DecisionTreeClassifier(max_depth = 3, criterion = 'log_loss', splitter = 'random')
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Estandarización
    ('ab', AdaBoostClassifier(
           random_state=42))  # voy a usar gradient boost
])



param_grid = {
    'ab__estimator':[tree1,tree2,tree3,tree4,tree5,tree6],
    'ab__n_estimators':[50,100,1000],
    'ab__learning_rate':[0.01,0.1,1],
    'ab__algorithm':['SAMME.R', 'SAMME']
    
}


grid_search = RandomizedSearchCV(
    estimator = pipeline,
    param_distributions = param_grid,
    scoring = 'accuracy',
    n_jobs = -1,
    n_iter = 100,
    cv=5
)



grid_search.fit(X_train,y_train)


results = grid_search.cv_results_
df = pd.DataFrame(results)
#df.columns
df_sorted = df.sort_values(by='rank_test_score', ascending=True)
df_top10 = df_sorted[df_sorted['rank_test_score'] <= 10]
pd.set_option('display.max_columns', None)

params = df_top10.params[20]
pipeline.set_params(**params)
pipeline.fit(X_train,y_train)

preds_train = pipeline.predict(X_train)
preds_test = pipeline.predict(X_test)

print(classification_report(y_train, preds_train))
print(classification_report(y_test, preds_test))
accuracy_score(y_test, preds_test)

