import pandas as pd

# fetch dataset
df = pd.read_csv("data/spambase.data", header=None)

# data (as pandas dataframes)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(df.loc[10])
