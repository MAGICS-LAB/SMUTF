import pandas as pd

df = pd.read_csv('./pair_0/Table1.csv')
df.reset_index(drop=True)
print(df.head())