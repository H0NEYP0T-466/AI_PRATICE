import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

model=LogisticRegression(max_iter=1000)
object=LabelEncoder()

data=pd.read_csv("survey lung cancer.csv")
copyData=data.copy()
copyData=copyData.replace({1:0, 2:1})
copyData["GENDER"]=object.fit_transform(copyData["GENDER"])
copyData["LUNG_CANCER"]=object.fit_transform(copyData["LUNG_CANCER"])

X=copyData[['AGE', 'GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING', 'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH', 'SWALLOWING DIFFICULTY', 'CHEST PAIN']]
y=copyData['LUNG_CANCER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model.fit(X_train, y_train)
prediction=model.predict(X_test)



print('Enter the Age:')
age = int(input())
print('Enter the Gender (1:Male/0:Female):')
gender = int(input())
print('Enter the Smoking Habit (1: Yes, 0: No):')
smoking = int(input())
print('Enter the Yellow Fingers (1: Yes, 0: No):')
yellow_fingers = int(input())
print('Enter the Anxiety Level (1: Yes, 0: No):')
anxiety = int(input())
print('Enter the Peer Pressure (1: Yes, 0: No):')
peer_pressure = int(input())
print('Enter the Chronic Disease (1: Yes, 0: No):')
chronic_disease = int(input())
print('Enter the Fatigue Level (1: Yes, 0: No):')
fatigue = int(input())
print('Enter the Allergy (1: Yes, 0: No):')
allergy = int(input())
print('Enter the Wheezing (1: Yes, 0: No):')
wheezing = int(input())
print('Enter the Alcohol Consumption (1: Yes, 0: No):')
alcohol_consuming = int(input())
print('Enter the Coughing (1: Yes, 0: No):')
coughing = int(input())
print('Enter the Shortness of Breath (1: Yes, 0: No):')
shortness_of_breath = int(input())
print('Enter the Swallowing Difficulty (1: Yes, 0: No):')
swallowing_difficulty = int(input())
print('Enter the Chest Pain (1: Yes, 0: No):')
chest_pain = int(input())

custom_data = pd.DataFrame({
    'AGE': [age],
    'GENDER': [gender],
    'SMOKING': [smoking],
    'YELLOW_FINGERS': [yellow_fingers],
    'ANXIETY': [anxiety],
    'PEER_PRESSURE': [peer_pressure],
    'CHRONIC DISEASE': [chronic_disease],
    'FATIGUE ': [fatigue],
    'ALLERGY ': [allergy],
    'WHEEZING': [wheezing],
    'ALCOHOL CONSUMING': [alcohol_consuming],
    'COUGHING': [coughing],
    'SHORTNESS OF BREATH': [shortness_of_breath],
    'SWALLOWING DIFFICULTY': [swallowing_difficulty],
    'CHEST PAIN': [chest_pain]
})

custom_prediction = model.predict(custom_data)
print("Custom Prediction:", custom_prediction)


print("Accuracy:", accuracy_score(y_test, prediction))
print("Precision:", precision_score(y_test, prediction))
print("Recall:", recall_score(y_test, prediction))
print("F1 Score:", f1_score(y_test, prediction))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, prediction))
print("\nClassification Report:\n", classification_report(y_test, prediction))


plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, prediction), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.figure(figsize=(8, 6))
sns.scatterplot(data=copyData, x="AGE", y="SMOKING", hue="LUNG_CANCER", palette="coolwarm", alpha=0.7)


plt.scatter(custom_data['AGE'], custom_data['SMOKING'], color='black', s=200, edgecolor="yellow", marker="X", label="Custom Input")
plt.title("Lung Cancer Dataset (Age vs Smoking) + Custom Input")
plt.legend()
plt.show()