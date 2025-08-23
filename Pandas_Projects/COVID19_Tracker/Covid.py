import numpy as np
import pandas as pd

# Load dataset
worldwide = pd.read_csv("worldometer_data.csv")

# Handle missing values
worldwide = worldwide.fillna(0)

# Global statistics
total_cases = np.sum(worldwide['TotalCases'])
total_deaths = np.sum(worldwide['TotalDeaths'])
total_recovered = np.sum(worldwide['TotalRecovered'])

print("=== 🌍 Worldwide COVID-19 Summary ===")
print(f"Total Registered Cases: {total_cases:,}")
print(f"Total Deaths: {total_deaths:,}")
print(f"Total Recovered: {total_recovered:,}")

# Asia-specific deaths
asia_deaths = np.sum(worldwide[worldwide['Continent'] == 'Asia']['TotalDeaths'])
print(f"\nTotal Deaths in Asia: {asia_deaths:,}")

# Pakistan-specific deaths
pakistan_deaths = np.sum(worldwide[worldwide['Country/Region'] == 'Pakistan']['TotalDeaths'])
print(f"Total Deaths in Pakistan: {pakistan_deaths:,}")

# Growth rate = NewCases / (TotalCases - NewCases)
growth_rate = (np.sum(worldwide['NewCases']) /
              (total_cases - np.sum(worldwide['NewCases']) + 1))
print(f"\nGlobal Growth Rate: {growth_rate:.4f}")

# Recovery rate = Recovered / Total Cases
recovery_rate = total_recovered / total_cases
print(f"Global Recovery Rate: {recovery_rate:.4f}")

# Top 5 countries with highest cases
print("\nTop 5 Countries by Total Cases:")
print(worldwide.nlargest(5, 'TotalCases')[['Country/Region', 'TotalCases']])

# Top infected continents
continent_cases = worldwide.groupby("Continent")["TotalCases"].sum().reset_index()
print("\nTop Infected Continents:")
print(continent_cases.nlargest(5, "TotalCases"))

print('the summary of this is:')
summary = {
    "Total Cases": np.sum(worldwide['TotalCases']),
    "Total Deaths": np.sum(worldwide['TotalDeaths']),
    "Total Recovered": np.sum(worldwide['TotalRecovered']),
    "Global Recovery Rate": np.sum(worldwide['TotalRecovered']) / np.sum(worldwide['TotalCases']),
}
print(pd.DataFrame([summary]))
