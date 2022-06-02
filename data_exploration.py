import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from data_preprocessing import get_data, preprocessing, pred_IstGkfz, prepare_number_of_accidents
from sklearn.preprocessing import StandardScaler

#Begrüßung.
print('\nWillkommen beim Data-Exploration Bereich des Accident Prediction Tools!')
print('\nEinlesen und Verarbeiten der Daten ...')

#Einlesen und Verarbeiten der Daten.
try:
    df_unfallatlas = pd.read_csv('data/df_unfallatlas.csv', delimiter=';', low_memory=False)
except:
    df_unfallatlas = get_data()
    df_unfallatlas = preprocessing(df_unfallatlas)
    df_unfallatlas = pred_IstGkfz(df_unfallatlas)

print('\nAnalyse der Daten ...')

#Analyse der Daten______________________________________________________________________________________________________

#Darstellung der Korrelation der Attribute.
#sns.set()
plt.figure(figsize=(15, 10))
plt.title('Korrelation der Attribute', fontsize=25, pad=20)
sns.heatmap(df_unfallatlas.corr(), annot=True, robust=True, fmt='.3f')
plt.show()

# Darstellung der Wochentage mit den meisten Unfällen pro Jahr.
weekdays = ['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag']
accweekday = df_unfallatlas.groupby(['UJAHR', 'UWOCHENTAG']).size()
accweekday = accweekday.rename_axis(['UJAHR', 'UWOCHENTAG']).unstack('UWOCHENTAG')
accweekday.columns = weekdays

plt.figure(figsize=(15, 10))
sns.heatmap(accweekday, cmap='plasma_r')
plt.title('Unfälle nach Wochentagen pro Jahr', fontsize=25, pad=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()


fahrzeuge_bayern = df_unfallatlas.loc[df_unfallatlas['ULAND'] == 9]
fahrzeuge_bayern = pd.DataFrame(fahrzeuge_bayern[[ 'IstRad', 'IstPKW', 'IstKrad', 'IstGkfz','IstSonstige', 'IstFuss']].sum(axis=0)).sort_values(by=0)
fahrzeuge_bayern = fahrzeuge_bayern.rename(columns={0:'Anzahl der Unfälle'})

fig, ax = plt.subplots(figsize = (15,10))
sns.barplot(x=fahrzeuge_bayern.index, y=fahrzeuge_bayern['Anzahl der Unfälle'])
plt.title('Anzahl der Unfälle pro Vekehrsmittel in Bayern', fontsize = 25, pad = 20)
plt.ylabel('')
ax.tick_params(axis='x', labelsize = 20)
ax.tick_params(axis='y', labelsize = 20)
plt.show()


bundesländer = ['SH', 'HH', 'NI', 'HB', 'NW',
                'HE', 'RP', 'BW', 'BY',
                'SL', 'BE','BB', 'MV', 'SN', 'ST', 'TH']


unfälle_bundesland = df_unfallatlas.groupby(df_unfallatlas['ULAND']).count()
unfälle_bundesland = pd.DataFrame(unfälle_bundesland['AGS']).rename(columns={'AGS':'Anzahl der Unfälle'})
unfälle_bundesland['Land'] = bundesländer
unfälle_bundesland = unfälle_bundesland.set_index(unfälle_bundesland['Land'])
unfälle_bundesland = unfälle_bundesland.drop(['Land'], axis=1)

fig, ax = plt.subplots(figsize = (15,10))
sns.barplot(x=unfälle_bundesland.index, y=unfälle_bundesland['Anzahl der Unfälle'])
plt.title('Anzahl der Unfälle pro Bundesland', fontsize = 25, pad = 20)
plt.ylabel('')
plt.xlabel('')
ax.tick_params(axis='x', labelsize = 20)
ax.tick_params(axis='y', labelsize = 20)
plt.show()


def corr_unfallbeteiligte(df_unfallatlas):
    data = pd.read_csv('data/exog_data/Externe_Unfalldaten_m.csv', sep = ';')
    data['day'] = 1
    data.rename(columns={'Jahr': 'year', 'Monat': 'month'},inplace=True)
    data = pd.DataFrame(data.set_index(pd.to_datetime(data[['year', 'month', 'day']])))
    data = data.drop(['year', 'month'], axis = 1)
    data = data.fillna(0)

    data_1 = data.loc[data['Fehlverhalten der Fahrzeugführer und Fußgänger'] == 1]
    data_1 = data_1.drop(['Fehlverhalten der Fahrzeugführer und Fußgänger'], axis = 1)
    data_1 = pd.DataFrame(data_1.sum(axis=1))
    data_2 = data.loc[data['Fehlverhalten der Fahrzeugführer und Fußgänger'] == 2]
    data_2 = data_2.drop(['Fehlverhalten der Fahrzeugführer und Fußgänger'], axis = 1)
    data_2 = pd.DataFrame(data_2.sum(axis=1))
    data_3 = data.loc[data['Fehlverhalten der Fahrzeugführer und Fußgänger'] == 3]
    data_3 = data_3.drop(['Fehlverhalten der Fahrzeugführer und Fußgänger'], axis = 1)
    data_3 = pd.DataFrame(data_3.sum(axis=1))


    # Erstellung des AGS.
    ags = '09162000'
    df_unfallatlas['AGS'] = df_unfallatlas['ULAND'].astype(str).str.zfill(2) \
                            + df_unfallatlas['UREGBEZ'].astype(str).str.zfill(1) \
                            + df_unfallatlas['UKREIS'].astype(str).str.zfill(2) \
                            + df_unfallatlas['UGEMEINDE'].astype(str).str.zfill(3)

    df_number_of_accidents = df_unfallatlas[(df_unfallatlas['AGS'] == ags)].reset_index(drop = True)
    df_number_of_accidents.rename(columns={'UJAHR': 'year', 'UMONAT': 'month', 'UWOCHENTAG': 'day', 'UKATEGORIE': 'Number of Accidents'},inplace=True)
    df_number_of_accidents = pd.DataFrame(df_number_of_accidents.set_index(pd.to_datetime(df_number_of_accidents[['year', 'month', 'day']])).resample('M')['Number of Accidents'].count())
    df_number_of_accidents.index = df_number_of_accidents.index.map(lambda t: t.replace(day = 1))
    df_number_of_accidents.index.freq = 'MS'

    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler()
    df_number_of_accidents['Anzahl der Unfälle'] = scaler.fit_transform(df_number_of_accidents)
    data_1[0] = scaler.fit_transform(data_1)
    data_2[0] = scaler.fit_transform(data_2)
    data_3[0] = scaler.fit_transform(data_3)

    corr = np.corrcoef(df_number_of_accidents['Anzahl der Unfälle'], data_3[0])

    fig, ax = plt.subplots(figsize = (15,10))
    sns.lineplot(x=df_number_of_accidents.index, y=df_number_of_accidents['Anzahl der Unfälle'])
    #sns.lineplot(x=data_1.index, y=data_1[0])
    sns.lineplot(x=data_2.index, y=data_2[0])
    #sns.lineplot(x=data_3.index, y=data_3[0])
    plt.title('test', fontsize = 25, pad = 20)
    plt.ylabel('')
    plt.xlabel('')
    ax.tick_params(axis='x', labelsize = 20)
    ax.tick_params(axis='y', labelsize = 20)
    plt.show()

#corr_unfallbeteiligte(df_unfallatlas)

df_unfallkategorie = df_unfallatlas.loc[df_unfallatlas['ULAND'] == 9]
df_unfallkategorie = df_unfallkategorie.groupby(df_unfallkategorie['UKATEGORIE']).count()
df_unfallkategorie = pd.DataFrame(df_unfallkategorie['AGS']).rename(columns={'AGS':'Anzahl der Unfälle'})

unfallkategorien = ['Unfall mit Getöteten', 'Unfall mit Schwerverletzten', 'Unfall mit Leichtverletzten']

df_unfallkategorie['Unfallkategorie'] = unfallkategorien
df_unfallkategorie = df_unfallkategorie.set_index(df_unfallkategorie['Unfallkategorie'])
df_unfallkategorie = df_unfallkategorie.drop(['Unfallkategorie'], axis=1)

fig, ax = plt.subplots(figsize = (15,10))
sns.barplot(x=df_unfallkategorie.index, y=df_unfallkategorie['Anzahl der Unfälle'])
plt.title('Anzahl der Unfälle pro Unfallkategorie in Bayern', fontsize = 25, pad = 20)
plt.ylabel('')
plt.xlabel('')
ax.tick_params(axis='x', labelsize = 18)
ax.tick_params(axis='y', labelsize = 20)
plt.show()

print()