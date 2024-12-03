# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 16:29:25 2024

@author: santi
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import math
os.chdir('Downloads/ap supervisado')

train_df = pd.read_csv('diabetes_prediction_dataset_train-labeled.csv')
print(train_df.shape)
print(train_df.describe())


train_df.columns

g = sns.FacetGrid(train_df[train_df['age'] < 30], col = 'diabetes',  sharey=False)

g.map_dataframe(sns.histplot, x='age', hue='diabetes', kde=True) ## lo primer que se ve es que esta super desbalanceado. 

g.add_legend()
plt.show()

g = sns.FacetGrid(train_df, col = 'gender', hue = 'diabetes',  sharey=True)
g.map(sns.histplot,'age', kde = True) ### no parece haber un efecto marcado del sexo
plt.show()


train_df['hypertension'].value_counts()
g = sns.FacetGrid(train_df, col = 'gender', row = 'hypertension', hue = 'diabetes',  sharey=False)
g.map_dataframe(sns.histplot,x ='age', bins = 15, kde = True) ### no parece haber un efecto marcado del sexo
plt.show()


g = sns.FacetGrid(train_df, col = 'heart_disease', hue = 'diabetes',  sharey=False)
#g.map(sns.histplot,'age', kde = True) ### no parece haber un efecto marcado del sexo
g.map_dataframe(sns.histplot,x ='age', bins = 15, kde = True) ### no parece haber un efecto marcado del sexo
plt.show() ## hay menos cantidad pero en general hay mas enfermos
## el efecto de heartdisease en la diabetes es notable


g = sns.FacetGrid(train_df, col = 'smoking_history', hue = 'diabetes',  sharey=False)
g.map(sns.histplot,'age', kde = True) ### no parece haber un efecto marcado del sexo
plt.show()


g = sns.FacetGrid(train_df, col = 'diabetes', hue = 'diabetes',  sharey=False, sharex = True)
g.map_dataframe(sns.histplot,x = 'blood_glucose_level', bins = 30, kde = True) ### no parece haber un efecto marcado del sexo
plt.show()


## boxplots #######
sns.boxplot(train_df, x = 'diabetes', y = 'age',hue = 'diabetes') ## en general la edad tiene un efecto en la diabetes
sns.boxplot(train_df, x = 'hypertension', y = 'age',hue = 'diabetes') ## aca no tenemos una idea de frecuencia. Se diluye el efecto 

sns.boxplot(train_df, x = 'diabetes', y = 'age',hue = 'gender') ## no parece haber un efecto del sexo
sns.boxplot(train_df, x = 'diabetes', y = 'blood_glucose_level',hue = 'diabetes')
train_df.columns
sns.boxplot(train_df, x = 'diabetes', y = 'HbA1c_level',hue = 'diabetes')

sns.boxplot(train_df, x = 'gender', y = 'age',hue = 'diabetes') ## visto de otra forma

sns.boxplot(train_df, x = 'heart_disease', y = 'age',hue = 'diabetes') ## similar a lo que pasa con la hipertension . 

sns.boxplot(train_df, x = 'smoking_history', y = 'age',hue = 'diabetes') 
 ## hay un leve efecto en aquellos que son current en bajar la edad de los que tienen diabetes
### 
train_df.columns


### blood glucose level

sns.boxplot(train_df, x = 'diabetes', y = 'blood_glucose_level',hue = 'diabetes') 
train_df['diabetes'].dtype


df_cont = train_df.loc[:,['bmi', 'HbA1c_level', 'blood_glucose_level','diabetes']] 

g = sns.PairGrid(df_cont, hue = 'diabetes')
g.map_diag(sns.histplot, bins = 10)
g.map_offdiag(sns.scatterplot)
g.add_legend()


def graph_group(df, cols, plot_type='hist', nrows=None, ncols=None):
    if nrows is None and ncols is None:
        nrows = math.ceil(len(cols)**0.5) # hermoso 
        ncols = math.ceil(len(cols) / nrows)
    elif nrows is None:
        nrows = math.ceil(len(cols) / ncols)
    elif ncols is None:
        ncols = math.ceil(len(cols) / nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 4*nrows))
    fig.subplots_adjust(hspace=0.8, wspace=0.4)

    axes = axes.flatten() if len(cols) > 1 else [axes]

    for i, column in enumerate(cols):
        if plot_type == 'box':
            sns.boxplot(data=df[column], ax=axes[i])
        elif plot_type == 'hist':
            sns.histplot(data=df, x=column, ax=axes[i], bins=50)
            #axes[i].ticklabel_format(style='plain', axis='x')
        
        axes[i].set_title(column)

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
    
    

graph_group(df_cont, df_cont.columns[:-1], plot_type = 'box')




### me gustaria repetir todo sacando las de 200 en adelante


df_blood200 = train_df[train_df['blood_glucose_level'] < 200]
df_blood200.blood_glucose_level
g = sns.FacetGrid(df_blood200, col = 'gender', row = 'hypertension', hue = 'diabetes',  sharey=False)
g.map(sns.histplot,'age', kde = True) ### no parece haber un efecto marcado del sexo
plt.show()


g = sns.FacetGrid(df_blood200, col = 'heart_disease', hue = 'diabetes',  sharey=False)
g.map(sns.histplot,'age', kde = True) ### no parece haber un efecto marcado del sexo
plt.show() ## hay menos cantidad pero en general hay mas enfermos
## el efecto de heartdisease en la diabetes es notable


g = sns.FacetGrid(df_blood200, col = 'smoking_history', hue = 'diabetes',  sharey=False)
g.map(sns.histplot,'age', kde = True) ### no parece haber un efecto marcado del sexo
plt.show()


g = sns.FacetGrid(df_blood200, col = 'diabetes', hue = 'diabetes',  sharey=False, sharex = True)
g.map_dataframe(sns.histplot,x = 'blood_glucose_level', bins = 5, kde = True) ### no parece haber un efecto marcado del sexo
plt.show()

df_cont_blood200 = df_blood200.loc[:,['bmi', 'HbA1c_level', 'blood_glucose_level','diabetes']] 

g = sns.PairGrid(df_cont_blood200, hue = 'diabetes')
g.map_diag(sns.histplot, bins = 10)
g.map_offdiag(sns.scatterplot)
g.add_legend()
