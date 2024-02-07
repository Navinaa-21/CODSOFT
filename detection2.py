#importing dataset
import numpy as np 
import pandas as pd 

#importing dataset
data = pd.read_csv('creditcard.csv')

print(data.head())
print(data.describe())

#putting feature values
y= data['Class']
X= data.drop(columns=['Class'],axis=1)

#splitting the data into train test 
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size= 0.3,random_state=0)

# fitting randomforest model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()

#from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20,criterion='entropy', random_state=0,max_depth=10)
classifier.fit(X_train,y_train)

#prediction
y_pred = classifier.predict(X_test)

fruad_det=classifier.predict_proba(X)[0][0]
print("Detection:",fruad_det)