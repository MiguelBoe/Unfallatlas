import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt


#Einlesen der Daten und Verknüpfung in einem DataFrame.
files = glob.glob('data/*.csv')

df_unfallatlas = pd.DataFrame()
for f in files:
    csv = pd.read_csv(f, delimiter= ';', low_memory = False)
    df_unfallatlas = pd.concat([df_unfallatlas, csv])


#Aufbereitung der Daten.
df_unfallatlas = df_unfallatlas.reset_index(drop = True)

df_unfallatlas['ULICHTVERH'] = df_unfallatlas['ULICHTVERH'].fillna(df_unfallatlas['LICHT'])
df_unfallatlas['STRZUSTAND'] = df_unfallatlas['STRZUSTAND'].fillna(df_unfallatlas['IstStrasse'])
df_unfallatlas['IstSonstige'] = df_unfallatlas['IstSonstige'].fillna(df_unfallatlas['IstSonstig'])

df_unfallatlas.drop(['FID', 'OBJECTID', 'OBJECTID_1', 'LICHT', 'IstStrasse', 'IstSonstig'], axis = 1, inplace = True)

df_unfallatlas = df_unfallatlas.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11,22,12,13,14,15,16,23,17,18,19,20,21,24]]

df_unfallatlas['AGS'] = df_unfallatlas['ULAND'].astype(str).str.zfill(2) \
                      + df_unfallatlas['UREGBEZ'].astype(str).str.zfill(1) \
                      + df_unfallatlas['UKREIS'].astype(str).str.zfill(2) \
                      + df_unfallatlas['UGEMEINDE'].astype(str).str.zfill(3)


#Ausgabe der Unfälle in Siegen im Jahr 2020.
df_unfallatlas_2020 = df_unfallatlas.loc[df_unfallatlas['UJAHR'] == 2020]
print('\nAnzahl der Unfälle in Siegen:', len(df_unfallatlas_2020.loc[df_unfallatlas_2020['AGS'] == '05970040']))


#Abfrage der Anzahl der Unfälle nach Jahr und AGS.
jahr = int(input('\nJahr: '))
ags = input('Gemeinde: ')
df_unfallatlas_year = df_unfallatlas.loc[df_unfallatlas['UJAHR'] == jahr]
print(f'\nAnzahl der Unfälle in {ags}:', len(df_unfallatlas_year.loc[df_unfallatlas_year['AGS'] == ags]))


df_unfallatlas_count = df_unfallatlas.groupby('AGS').size().reset_index(name = 'number_of_accidents')
df_unfallatlas_count = df_unfallatlas_count.sort_values('number_of_accidents', ascending = False)

sns.set()
fig, ax = plt.subplots(figsize = (15,10))
sns.barplot(x = 'AGS', y = 'number_of_accidents', data = df_unfallatlas_count.head(10))
plt.title('Die 10 Gemeinden mit den meisten Unfällen von 2016 - 2020', fontsize = 30, pad = 20)
plt.ylabel('Anzahl der Unfälle', fontsize = 25)
plt.xlabel('AGS', fontsize = 25)
ax.tick_params(axis = 'x', labelsize = 15)
ax.tick_params(axis = 'y', labelsize = 20)
plt.show()


