import numpy as np
import pandas as pd

country_wise = pd.read_csv("country_wise_latest.csv")
worldwide=pd.read_csv("worldometer_data.csv")

print(country_wise.head(10))
print(worldwide.tail(10))