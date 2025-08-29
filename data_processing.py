import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler

data=pd.read_csv('student_dataset.csv')
copy=data.copy()

labelencoder=LabelEncoder()
copy['Gender']=labelencoder.fit_transform(copy['Gender'])
copy['Passed']=labelencoder.fit_transform(copy['Passed'])

print(copy[['Gender','Passed']].head(10))

#one hot encoding on Year
copy_encoded=pd.get_dummies(copy,columns=['Year'],dtype=int)
print("one hot encoding")
print(copy_encoded.head(10))

#feature scaling

data={
    "studyHours":[1,2,3,4,5],
    "marks":[120,150,170,200,250]
}
objectofstandardscaler=StandardScaler()
objectofmixmax=MinMaxScaler()


new_data=pd.DataFrame(data)
standardvalue=objectofstandardscaler.fit_transform(new_data)
minmaxvalue=objectofmixmax.fit_transform(new_data)
print("standard value is:")
print(standardvalue)
print("min max value is:")
print(minmaxvalue)