import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


data = {
    "studyHours": [1, 2, 3, 4, 5],
    "marks": [120, 150, 10, 200, 250]
}

df = pd.DataFrame(data)
X = df[["studyHours"]]
y = df["marks"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

hours = float(input("Enter the number of study hours: "))


model = LinearRegression()
model.fit(X_train, y_train)


predicted_marks = model.predict(pd.DataFrame([[hours]], columns=["studyHours"]))
print(f"Predicted marks for {hours} study hours: {predicted_marks[0]}")

plt.scatter(X, y, color="blue", label="Actual Data")         
plt.plot(X,model.predict(X), color="red", label="Regression Line") 


plt.scatter(hours, predicted_marks, color="green", s=100, label="Predicted Point")

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Linear Regression - Study Hours vs Marks")
plt.legend()
plt.show()
