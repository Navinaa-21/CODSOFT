#importing libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

#importing dataset
df = pd.read_csv('spam.csv')

#visualizing the dataset
print(df.columns())
print(df.tail())

#putting feature value to x
x = df.drop('Amount',axis=1)
y = df['Amount']

#Train-Test-Split the data
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=42)
x_train.shape, x_test.shape

from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(random_state=42, n_jobs=-1, max_depth=5, n_estimators=100, oob_score=True)
classifier_rf.fit(x_train,y_train)
classifier_rf.oob_score_
