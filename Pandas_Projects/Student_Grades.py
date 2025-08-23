import pandas as p
import numpy as np


data=p.read_csv("Student_Grades.csv")
# print(data)

print("the total student in the class are:")
print(len(data))
print("the average marks in this class is:")
print(data['Scores'].mean())
print("the top performer is:")
print(data.loc[data['Scores'].idxmax()])
print("The number of student passed are")
print(np.sum((data['Grade'] =='A') | (data['Grade'] =='B')))
print("the number of student failed are:")
print(np.sum(data['Grade'] == 'C'))
print("the fail rates are:")
print(np.sum(data['Grade'] == 'C') / len(data) * 100, "%")
print("the pass rates are:")
print(np.sum((data['Grade'] =='A') | (data['Grade'] =='B')) / len(data) * 100, "%")
