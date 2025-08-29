import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.preprocessing import LabelEncoder

data=pd.read_csv('student_dataset.csv')
copy=data.copy()

labelencoder=LabelEncoder()
copy['Gender']=labelencoder.fit_transform(copy['Gender'])
copy['Passed']=labelencoder.fit_transform(copy['Passed'])

print(copy[['Gender','Passed']].head(10))