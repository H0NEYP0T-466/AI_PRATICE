import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

data = {
    "studyHours": [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
        11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        21, 22, 23, 24, 25, 26, 27, 28, 29, 30
    ],
    "marks": [
        58, 121, 148, 184, 262, 277, 340, 408, 477, 488,
        533, 612, 641, 717, 751, 797, 833, 908, 944, 1003,
        1072, 1071, 1143, 1213, 1249, 1307, 1321, 1429, 1440, 1502
    ]
}


df = pd.DataFrame(data)
X = df[["studyHours"]]
y = df["marks"]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

hours = float(input("Enter the number of study hours: "))


model = LinearRegression()
model.fit(X_train, y_train)


predicted_marks = model.predict(pd.DataFrame([[hours]], columns=["studyHours"]))
print(f"Predicted marks for {hours} study hours: {predicted_marks[0]}")

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print("R² Score:", r2)

plt.scatter(X, y, color="blue", label="Actual Data")         
plt.plot(X,model.predict(X), color="red", label="Regression Line") 


plt.scatter(hours, predicted_marks, color="green", s=100, label="Predicted Point")

plt.xlabel("Study Hours")
plt.ylabel("Marks")
plt.title("Linear Regression - Study Hours vs Marks")
plt.legend()
plt.show()
