#!/usr/bin/env python3.11

import pandas as pd

df = pd.read_excel('Inkindo Model.xlsx', header=[0,1], index_col=0)
df = df.stack([0, 1])
df = df.reset_index()
df.columns = ['Experience', 'Year', 'Degree', 'Billing']
df['Experience'] = df['Experience'].astype(pd.UInt8Dtype())
df['Year'] = df['Year'].astype(pd.UInt16Dtype())
df['Degree'] = df['Degree'].astype(
	pd.CategoricalDtype(['Bachelor', 'Master', 'Doctorate'], True)
)
df['Billing'] = df['Billing'].astype(pd.UInt32Dtype())
df = df[['Year', 'Degree', 'Experience', 'Billing']]
df = df.sort_values(list(df.columns))

print(df)
print(df.info())
print(df.index)
print(df.columns)

# print(df['Billing'].to_numpy(dtype='uint64').dtype)
# print(df[(df['Degree'] == 'Bachelor') & (df['Experience'] == 3)])

df.to_csv('Inkindo.csv', index=None)

print('Hello, world!')
