import pandas as pd
import glob

files = glob.glob('data/*.csv')

df_unfallatlas = pd.DataFrame()
for f in files:
    csv = pd.read_csv(f, delimiter= ';', nrows = 10)
    df_unfallatlas = pd.concat([df_unfallatlas, csv])


df_unfallatlas['ULICHTVERH'] = df_unfallatlas['ULICHTVERH'].fillna(df_unfallatlas['LICHT'])
df_unfallatlas['STRZUSTAND'] = df_unfallatlas['STRZUSTAND'].fillna(df_unfallatlas['IstStrasse'])
df_unfallatlas['IstSonstige'] = df_unfallatlas['IstSonstige'].fillna(df_unfallatlas['IstSonstig'])

df_unfallatlas.drop(['FID', 'OBJECTID', 'OBJECTID_1', 'LICHT', 'IstStrasse', 'IstSonstig'], axis = 1, inplace = True)

df_unfallatlas = df_unfallatlas.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,22,12,13,14,15,16,23,17,18,19,20,21,24]]

