# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 18:32:45 2024

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
from sklearn.model_selection import GridSearchCV, train_test_split



train_df = pd.read_csv('diabetes_prediction_dataset_train-labeled.csv')

### pipelinde trabajo 
y = train_df['diabetes']
X = train_df.iloc[:,:-1]

## 
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
## division del set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

## pipline standard scaler
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Estandarización
    ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))  # vot a usar random forestt
])

## defino los hiperparametros a explorar en el gridsearch
param_grid = {
    'rf__n_estimators': [50, 100, 1000],
    'rf__max_features': ['sqrt', 'log2'],
    'rf__max_depth' : [10, 20, None],
    'rf__criterion' :['gini', 'entropy'],
    'rf__class_weight':['balanced','None']
}

## defino la busqueda
grid_search = GridSearchCV(
    estimator = pipeline,
    param_grid = param_grid,
    scoring = 'f1_macro',
    cv=5
)

## ajusto los modelos
grid_search.fit(X_train, y_train) 


results = grid_search.cv_results_
df = pd.DataFrame(results)
#df.columns
df_sorted = df.sort_values(by='rank_test_score', ascending=True)
df_top10 = df_sorted[df_sorted['rank_test_score'] <= 10]
df_top10.columns
pd.set_option('display.max_columns', None)
df_top10[['param_rf__criterion', 'param_rf__n_estimators','param_rf__max_depth', 'param_rf__max_features', 'param_rf__class_weight', 'mean_test_score', 'std_test_score']]



params_dict = df_top10.loc[35, 'params']
pipeline.set_params(**params_dict)

pipeline.fit(X_train, y_train)


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, roc_curve, roc_auc_score
y_pred =  pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
print(classification_report(y_train,y_pred))
print(classification_report(y_test,y_pred_test))
print(accuracy_score(y_test,y_pred_test))
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
auc_score = roc_auc_score(y_test, y_pred_prob)
print(f"AUC del modelo seleccionado: {auc_score}")

specificity = 1 - fpr
plt.figure()
plt.plot(tpr, specificity, color='red', lw=2, label=f'Curva ROC (área = {auc_score:.2f})')
plt.plot([0, 1], [1, 0], color='blue', linestyle='--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Especificidad')
plt.ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
plt.title('Curva ROC')
plt.legend(loc="lower right")
plt.show()

## el threshold que maximiza especificidad y sencibilidad
# Calcular la métrica de Youden
J = tpr - fpr ## maximizo este valor
optimal_threshold_index = J.argmax()
optimal_threshold = thresholds[optimal_threshold_index]
optimal_threshold ## este es el ubmbral que maximiza la especificidad y la sencibilidad, es el que deberia usarse para establecer la linea de corte y es 0.4698
print('el umbrol optimo es', optimal_threshold)


y_pred_test_optimal = (y_pred_prob >= optimal_threshold).astype(int)

print(classification_report(y_test,y_pred_test_optimal)) ### tiene muchos problemas con el 1 



## gboost

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Estandarización
    ('gb', GradientBoostingClassifier(loss='log_loss', max_depth=10, n_estimators = 100,random_state=42))  # vot a usar gradient boost
])

pipeline.fit(X_train, y_train)
y_pred =  pipeline.predict(X_train)
y_pred_test = pipeline.predict(X_test)
y_pred_prob = pipeline.predict_proba(X_test)[:, 1]
print(classification_report(y_train,y_pred))
print(classification_report(y_test,y_pred_test))


### voy aconsiderar el balnaceo de los datos 
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
len(X_train_under)


pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Estandarización
    ('gb', GradientBoostingClassifier(loss='log_loss', max_depth=10, n_estimators = 100,random_state=42))  # vot a usar gradient boost
])

pipeline.fit(X_train_under, y_train_under)

y_pred_under =  pipeline.predict(X_train_under)
y_pred_test= pipeline.predict(X_test)
print(classification_report(y_train_under,y_pred_under))
print(classification_report(y_test,y_pred_test))


## sobre muestreo 
def oversample(X, y, target_name):
    data = pd.concat([X, y], axis=1)
    majority_class = data[data[target_name] == 0]
    minority_class = data[data[target_name] == 1]
    minority_upsampled = resample(minority_class, replace=True, n_samples=len(majority_class)/2, random_state=42)
    balanced_data = pd.concat([majority_class, minority_upsampled])
    return balanced_data.drop(target_name, axis=1), balanced_data[target_name]

X_train_over, y_train_over = oversample(X_train, y_train, 'diabetes')
pipeline.fit(X_train_over, y_train_over)

y_pred_over =  pipeline.predict(X_train_over)
y_pred_test= pipeline.predict(X_test)

print(classification_report(y_train_over,y_pred_over))
print(classification_report(y_test,y_pred_test))

## smote
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)

X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
pipeline.fit(X_train_sm, y_train_sm)

y_pred_sm =  pipeline.predict(X_train_sm)
y_pred_test= pipeline.predict(X_test)

print(classification_report(y_train_sm,y_pred_sm))
print(classification_report(y_test,y_pred_test))
### sigue seleccion de variables, muy interesante el resultado de el sintetico 


pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Estandarización
    ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))  # vot a usar random forestt
])

pipeline.set_params(**params_dict)

pipeline.fit(X_train_sm, y_train_sm)

y_pred_sm =  pipeline.predict(X_train_sm)
y_pred_test= pipeline.predict(X_test)
print(classification_report(y_train_sm,y_pred_sm))
print(classification_report(y_test,y_pred_test))


### tengo que seleccionar variables


## arbol de desicion para seleccionar variables
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


s_model = DecisionTreeClassifier(random_state=0) ### voy a dejarlo por defecto

s_model.fit(X,y) ## uso completo por que quiero hacer seleccion de variables

fig = plt.figure(figsize=(25,20))
# add title to the plot
fig.suptitle(' Tree Classifier\n selecting variables', fontsize=50)
_ = tree.plot_tree(s_model, 
                   feature_names=X.columns,
                   filled=True)

importances = s_model.feature_importances_

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)


plt.figure(figsize=(12, 8))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances from Decision Tree')
plt.gca().invert_yaxis()  # To display the most important features at the top
plt.show()

### me pregunto si usando menos variables no podria mejorar. (estamos modelando ruido? 


## me pregunto si esto cambia codificando diferente smooking history

if train_df['smoking_history'].dtype == 'object':
    smoking_history_dict = {'No Info': 0, 'never': 1, 'ever': 2, 'former': 3, 'not current': 4, 'current': 5 }
    train_df['smoking_history'] = train_df['smoking_history'].map(smoking_history_dict)
    

y = train_df['diabetes']
X = train_df.iloc[:,:-1]

## 
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

s_model = DecisionTreeClassifier(random_state=0) ### voy a dejarlo por defecto
s_model.fit(X,y)

importances = s_model.feature_importances_

feature_importances = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)


plt.figure(figsize=(12, 8))
plt.barh(feature_importances['Feature'], feature_importances['Importance'], color='skyblue')
plt.xlabel('Importance')
plt.title('Feature Importances from Decision Tree')
plt.gca().invert_yaxis()  # To display the most important features at the top
plt.show()

## ahora si pienso que podria sacar varias 
X.columns
X_subdata = X.loc[:,['age', 'smoking_history', 'bmi', 'HbA1c_level', 'blood_glucose_level']]

X_train, X_test, y_train, y_test = train_test_split(X_subdata, y, test_size=0.2, random_state=42)
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Estandarización
    ('gb', GradientBoostingClassifier(loss='log_loss', max_depth=10, n_estimators = 1000,random_state=42))  # vot a usar gradient boost
])




smote = SMOTE(random_state=42)

X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
pipeline.fit(X_train_sm, y_train_sm)


y_pred =  pipeline.predict(X_train_sm)
y_pred_test= pipeline.predict(X_test)

print(classification_report(y_train_sm,y_pred))
print(classification_report(y_test,y_pred_test))


## voy a probar ahora con over sampling
def oversample(X, y, target_name):
    data = pd.concat([X, y], axis=1)
    majority_class = data[data[target_name] == 0]
    minority_class = data[data[target_name] == 1]
    minority_upsampled = resample(minority_class, replace=True, n_samples=int(len(majority_class/2)), random_state=42)
    balanced_data = pd.concat([majority_class, minority_upsampled])
    return balanced_data.drop(target_name, axis=1), balanced_data[target_name]

X_train_over, y_train_over = oversample(X_train, y_train, 'diabetes')
pipeline.fit(X_train_over, y_train_over)

y_pred_over =  pipeline.predict(X_train_over)
y_pred_test= pipeline.predict(X_test)

print(classification_report(y_train_over,y_pred_over))
print(classification_report(y_test,y_pred_test))


