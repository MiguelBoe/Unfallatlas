import pandas as pd

df_2016 = pd.read_csv('data/Unfallorte2016_LinRef.csv', delimiter= ';')
df_2017 = pd.read_csv('data/Unfallorte2017_LinRef.csv', delimiter= ';')
df_2018 = pd.read_csv('data/Unfallorte2018_LinRef.csv', delimiter= ';')
df_2019 = pd.read_csv('data/Unfallorte2019_LinRef.csv', delimiter= ';')
df_2020 = pd.read_csv('data/Unfallorte2020_LinRef.csv', delimiter= ';')

print(df_2016)