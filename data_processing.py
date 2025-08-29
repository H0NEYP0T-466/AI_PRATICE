import pandas as pd
import numpy as np
import sklearn as sk
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


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
copyofdata=new_data.copy()
standardvalue=objectofstandardscaler.fit_transform(copyofdata)
minmaxvalue=objectofmixmax.fit_transform(copyofdata)
print("standard value is:")
print(standardvalue)
print("min max value is:")
print(minmaxvalue)

X=copyofdata[["studyHours"]]
y=copyofdata[["marks"]]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("X-train data is:",X_train)
print("X-test data is:",X_test)
print("y-train data is:",y_train)
print("y-test data is:",y_test)