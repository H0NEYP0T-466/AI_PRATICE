import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("\n🏠 House Price Prediction")
house = pd.read_csv("house_data.csv")
X_house = house[["size", "rooms", "location_factor"]]
y_house = house["price"]


Xh_train, Xh_test, yh_train, yh_test = train_test_split(X_house, y_house, test_size=0.2, random_state=42)

model_house = LinearRegression()
model_house.fit(Xh_train, yh_train)

pred_house = model_house.predict(Xh_test)

size = int(input("Enter house size (sqft): "))
rooms = int(input("Enter number of rooms: "))
location_factor = int(input("Enter location factor (1-5): "))
custom_pred = model_house.predict([[size, rooms, location_factor]])[0]
print(f"Custom House Prediction: {custom_pred}")

print("R2 Score:", r2_score(yh_test, pred_house))
plt.scatter(X_house["size"], y_house, color="blue", label="Actual")
plt.scatter(Xh_test["size"], pred_house, color="red", label="Predicted")
plt.scatter(size, custom_pred, color="black", s=100, marker="x", label="Custom Prediction")
plt.xlabel("Size (sqft)"); plt.ylabel("Price"); plt.title("House Price Prediction")
plt.legend()
plt.savefig("house_price_prediction.png")
plt.show()




