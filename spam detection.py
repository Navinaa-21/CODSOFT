import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords
from collections import Counter

a=pd.read_csv("spam.csv",encoding = 'latin-1')
print(a.info())
a.isnull().sum()
a.drop(columns=a[['Unnamed: 2','Unnamed: 3','Unnamed: 4']],axis=1,inplace=True)
a=a.drop_duplicates()
a.columns=['spam/ham','sms']

a.loc[a['spam/ham'] == 'spam', 'spam/ham',] = 0
a.loc[a['spam/ham'] == 'ham', 'spam/ham',] = 1

x=a.sms
y =a['spam/ham']

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=3)
feat_vect=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
print(feat_vect)

ytrain=ytrain.astype('int')
ytest=ytest.astype('int')
xtrain_vec =feat_vect.fit_transform(xtrain)
xtest_vec =feat_vect.transform(xtest)

logi=LogisticRegression()
logi.fit(xtrain_vec,ytrain)

pred_logi=logi.predict(xtest_vec)
print(pred_logi)
print(accuracy_score(ytest,pred_logi))