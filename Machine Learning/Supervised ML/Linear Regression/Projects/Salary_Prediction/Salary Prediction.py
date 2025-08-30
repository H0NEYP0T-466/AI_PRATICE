import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score



print("\n💼 Salary Prediction")
salary = pd.read_csv("salary_data.csv")
X_sal = salary[["experience", "education_level"]]
y_sal = salary["salary"]


X_train, X_test, y_train, y_test = train_test_split(X_sal, y_sal, test_size=0.2, random_state=42)
model_salary = LinearRegression().fit(X_train, y_train)
pred_salary = model_salary.predict(X_test)


print("R2 Score:", r2_score(y_test, pred_salary))


experience = int(input("Enter years of experience: "))
education_level = int(input("Enter education level (1-4): "))
custom_pred_sal = model_salary.predict([[experience, education_level]])[0]
print(f"Custom Salary Prediction: {custom_pred_sal}")


plt.scatter(salary["experience"], y_sal, color="blue", label="Actual")
plt.scatter(X_test["experience"], pred_salary, color="red", label="Predicted")
plt.scatter(experience, custom_pred_sal, color="black", s=100, marker="x", label="Custom Prediction")
plt.xlabel("Experience (Years)"); plt.ylabel("Salary"); plt.title("Salary Prediction")
plt.legend()
plt.savefig("salary_prediction.png")
plt.show()