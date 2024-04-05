import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split


import warnings
warnings.filterwarnings('ignore')

data = pd.read_csv('Zomato_df.csv')
# print(data)

data.drop('Unnamed: 0',axis=1,inplace=True)  # Droping the unnamed column from our dataset
# print(data)

x = data.drop('rate',axis=1)
y = data['rate']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.30,random_state=12)

ET_model = ExtraTreesRegressor(n_estimators=120)
ET_model.fit(x_train,y_train)
y_pred = ET_model.predict(x_test)


import pickle 
# Saving model to disk
pickle.dump(ET_model, open('model.pkl','wb'))
model=pickle.load(open('model.pkl','rb'))
print(y_pred)