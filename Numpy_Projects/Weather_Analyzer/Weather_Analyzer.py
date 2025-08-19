import numpy as np

data = np.genfromtxt("weather_data.csv", 
                     delimiter=",", 
                     dtype=None,    
                     encoding="utf-8", 
                     names=True)     

location=data["Location"]
temperature=data["Temperature_C"]
humidity=data["Humidity_pct"]
precipitation=data["Precipitation_mm"]
wind_speed=data["Wind_Speed_kmh"]
date=data["Date_Time"]

print("the average temprature is",np.mean(temperature))
print("the average humidity is",np.mean(humidity))
print("the average precipitation is",np.mean(precipitation))
print("the average wind speed is",np.mean(wind_speed))

print("the maximum temprature was",np.max(temperature))
print("the maximum humidity was",np.max(humidity))
print("the maximum precipitation was",np.max(precipitation))
print("the maximum wind speed was",np.max(wind_speed))

print("the minimum temprature was",np.min(temperature))
print("the minimum humidity was",np.min(humidity))
print("the minimum precipitation was",np.min(precipitation))
print("the minimum wind speed was",np.min(wind_speed))

print(np.unique(location))


mask = location == "Chicago"
print("the maximum temprature in chicago was",np.max(temperature[mask]))
print("the minimum temprature in chicago was",np.min(temperature[mask]))
print("the average temprature in chicago was",np.mean(temperature[mask]))
print("the days with humidity with more than 80% were",np.sum(humidity[mask]>80))
print("the average temprature in chicago in F is",np.mean(temperature[mask]*9/5+32))

mask = location == "New York"
print("the maximum temprature in new york was",np.max(temperature[mask]))
print("the minimum temprature in new york was",np.min(temperature[mask]))
print("the average temprature in new york was",np.mean(temperature[mask]))
print("the days with humidity with more than 80% were",np.sum(humidity[mask]>80))
print("the average temprature in new york in F is",np.mean(temperature[mask]*9/5+32))


mask = location == "Los Angeles"
print("the maximum temprature in los angeles was",np.max(temperature[mask]))
print("the minimum temprature in los angeles was",np.min(temperature[mask]))
print("the average temprature in los angeles was",np.mean(temperature[mask]))
print("the days with humidity with more than 80% were",np.sum(humidity[mask]>80))
print("the average temprature in los angeles in F is",np.mean(temperature[mask]*9/5+32))


mask = location == "Dallas"
print("the maximum temprature in Dallas was",np.max(temperature[mask]))
print("the minimum temprature in Dallas was",np.min(temperature[mask]))
print("the average temprature in Dallas was",np.mean(temperature[mask]))
print("the days with humidity with more than 80% were",np.sum(humidity[mask]>80))
print("the average temprature in Dallas in F is",np.mean(temperature[mask]*9/5+32))


mask = location == "Houston"
print("the maximum temprature in Houston was",np.max(temperature[mask]))
print("the minimum temprature in Houston was",np.min(temperature[mask]))
print("the average temprature in Houston was",np.mean(temperature[mask]))
print("the days with humidity with more than 80% were",np.sum(humidity[mask]>80))
print("the average temprature in Houston in F is",np.mean(temperature[mask]*9/5+32))


mask = location == "Phoenix"
print("the maximum temprature in Phoenix was",np.max(temperature[mask]))
print("the minimum temprature in Phoenix was",np.min(temperature[mask]))
print("the average temprature in Phoenix was",np.mean(temperature[mask]))
print("the days with humidity with more than 80% were",np.sum(humidity[mask]>80))
print("the average temprature in Phoenix in F is",np.mean(temperature[mask]*9/5+32))

mask = location == "Philadelphia"
print("the maximum temprature in Philadelphia was",np.max(temperature[mask]))
print("the minimum temprature in Philadelphia was",np.min(temperature[mask]))
print("the average temprature in Philadelphia was",np.mean(temperature[mask]))
print("the days with humidity with more than 80% were",np.sum(humidity[mask]>80))
print("the average temprature in Philadelphia in F is",np.mean(temperature[mask]*9/5+32))




mask = location == "San Antonio"
print("the maximum temprature in San Antonio was",np.max(temperature[mask]))
print("the minimum temprature in San Antonio was",np.min(temperature[mask]))
print("the average temprature in San Antonio was",np.mean(temperature[mask]))
print("the days with humidity with more than 80% were",np.sum(humidity[mask]>80))
print("the average temprature in San Antonio in F is",np.mean(temperature[mask]*9/5+32))

mask = location == "San Diego"
print("the maximum temprature in San Diego was",np.max(temperature[mask]))
print("the minimum temprature in San Diego was",np.min(temperature[mask]))
print("the average temprature in San Diego was",np.mean(temperature[mask]))
print("the days with humidity with more than 80% were",np.sum(humidity[mask]>80))
print("the average temprature in San Diego in F is",np.mean(temperature[mask]*9/5+32))


mask = location == "San Jose"
print("the maximum temprature in San Jose was",np.max(temperature[mask]))
print("the minimum temprature in San Jose was",np.min(temperature[mask]))
print("the average temprature in San Jose was",np.mean(temperature[mask]))
print("the days with humidity with more than 80% were",np.sum(humidity[mask]>80))   
print("the average temprature in San Jose in F is",np.mean(temperature[mask]*9/5+32))


print("the hottest day overall was",date[np.argmax(temperature)],"with a temprature of",np.max(temperature))

print("the coldest day overall was",date[np.argmin(temperature)],"with a temprature of",np.min(temperature))

date_np = date.astype("datetime64[ns]") 
in_2024 = (date_np >= np.datetime64("2024-01-01")) & (date_np < np.datetime64("2025-01-01"))

custom_mask = (location == "New York") & in_2024
print("the average temprature in New York in 2024 was",np.mean(temperature[custom_mask]))





