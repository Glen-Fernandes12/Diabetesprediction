# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 20:16:24 2022

@author: Glen Fernandes
"""

import pandas as pd
import numpy as np
import pickle
# pickle is used to store in another file and read somwhere else

df =pd.read_csv("C:\\Users\\Glen Fernandes\\Desktop\\diabetes.csv")
#prediction
df.rename(columns={'DiabetesPedigreeFunction':'DPF'})# replacing the name
df_copy = df.copy(deep= True)
df_copy[['Glucose', 'BloodPressure' , 'SkinThickness','Insulin','BMI']]= df_copy[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
df_copy['Glucose'].fillna(df_copy['Glucose'].mean() ,inplace= True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean() ,inplace= True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].mean() ,inplace= True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median() ,inplace= True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace= True)

from sklearn.model_selection import train_test_split
X=df.drop(columns='Outcome')
y=df['Outcome']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20)

classifier.fit(X_train,y_train)

filename = 'diabetes.pdl'
pickle.dump(classifier ,open(filename ,'wb'))





