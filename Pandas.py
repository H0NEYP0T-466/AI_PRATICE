import pandas as p

data=p.Series([1, 2, 3, 4, 5],index=['a','b','c','d','e'])
print(data)

data = {
    'Name': ['Ali', 'Umer', 'Bilal'],
    'Age': [20, 22, 19],
    'Marks': [88, 92, 79]
}
df = p.DataFrame(data)
print(df)


data=p.read_csv("data.csv",encoding="latin1")
print(data)

