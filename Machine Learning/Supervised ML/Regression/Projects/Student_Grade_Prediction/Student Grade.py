import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


print("\n📚 Student Marks Prediction")
students = pd.read_csv("student_data.csv")
X_stud = students[["study_hours", "attendance", "assignments"]]
y_stud = students["marks"]


Xs_train, Xs_test, ys_train, ys_test = train_test_split(X_stud, y_stud, test_size=0.2, random_state=42)
model_student = LinearRegression().fit(Xs_train, ys_train)
pred_student = model_student.predict(Xs_test)

study_hours = int(input("Enter study hours: "))
attendance = int(input("Enter attendance percentage: "))
assignments = int(input("Enter assignment score: "))
custom_pred_stud = model_student.predict([[study_hours, attendance, assignments]])[0]
print(f"Custom Student Marks Prediction: {custom_pred_stud}")

print("R2 Score:", r2_score(ys_test, pred_student))
plt.scatter(students["study_hours"], y_stud, color="blue", label="Actual")
plt.scatter(Xs_test["study_hours"], pred_student, color="red", label="Predicted")
plt.scatter(study_hours, custom_pred_stud, color="black", s=100, marker="x", label="Custom Prediction")
plt.xlabel("Study Hours"); plt.ylabel("Marks"); plt.title("Student Marks Prediction")
plt.legend()
plt.savefig("student_marks_prediction.png")
plt.show()