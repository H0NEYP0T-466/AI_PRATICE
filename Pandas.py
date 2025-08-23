import pandas as p

data=p.Series([1, 2, 3, 4, 5],index=['a','b','c','d','e'])
print(data)

data = {
    'Name': ['Ali', 'Umer', 'Bilal'],
    'Age': [20, 22, 19],
    'Marks': [88, 92, 79]
}

maindata=p.read_csv("data.csv",encoding="latin1")

df = p.DataFrame(data)
print(df)
print("----")
print(df.loc[0])  # First row
print("----")
print(df.loc[1, 'Name'])
print("----")
print(df.loc[1:2])
print("CSV DATA")
# print(maindata.loc[:, 'Engines'])
print(maindata.shape)     # (10, 11) → 10 rows, 11 columns
print(maindata.columns)   # all column names
print(maindata.index)     # row index (0–9)
print(maindata.dtypes)    # data types of each column

print(maindata.loc[0])                 # first row (by label)
print(maindata.iloc[0, 1])             # first row, 2nd column → "SF90 STRADALE"
print(maindata.loc[:, ["Company Names","Cars Names","Engines"]])  # only these cols
print(maindata.iloc[0:3, 0:3])         # first 3 rows, first 3 cols


print('----')
maindata=maindata.dropna()  # drop rows with any missing values
print(maindata)
print("---")
print(maindata.isnull().sum())  # check for missing values
