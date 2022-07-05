import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from data_preprocessing import get_data, preprocessing, pred_IstGkfz, prepare_number_of_accidents
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter('ignore', FutureWarning)

#Begrüßung.
print('\nWillkommen beim Data-Exploration Bereich des Accident Prediction Tools!')
print('\nEinlesen und Verarbeiten der Daten ...')

'''
Im folgenden Abschnitt werden die Unfalldaten eingelesen. Dabei wird mit Exception Handling geprüft, ob bereits eine auf-
bereitete CSV-Datei (df_unfallatlas.csv) mit den Daten vorhanden ist. Ist dies der Fall, wird diese Datei einfach eingelesen.
Wenn nicht, werden die einzelnen CSV-Dateien mit den Rohdaten eingelesen und dann mit verschiedenen Funktionen aufbereitet. 
Diese Funktionen befinden sich in der Datei data_preprocessing und werden dort beschrieben.
'''
#Einlesen und Verarbeiten der Daten.
try:
    df_unfallatlas = pd.read_csv('data/df_unfallatlas.csv', delimiter=';', low_memory=False)
except:
    df_unfallatlas = get_data()
    df_unfallatlas = preprocessing(df_unfallatlas)
    df_unfallatlas = pred_IstGkfz(df_unfallatlas)

print('\nAnalyse der Daten ...')


'''
In dem Abschnitt "Analyse der Daten" werden einige Plots generiert, welche den Datensatz beschreiben.
'''
#Analyse der Daten______________________________________________________________________________________________________


#Darstellung der Korrelation der Attribute.
def correlation_heatmap(df_unfallatlas):
    sns.set_style('whitegrid')
    plt.figure(figsize=(15, 10))
    plt.title('Korrelation der Attribute', fontsize=25, pad=20)
    ax=sns.heatmap(df_unfallatlas.corr(), annot=True, robust=True, fmt='.3f')
    ax.figure.tight_layout()
    plt.show()
correlation_heatmap(df_unfallatlas)


#Darstellung der Anzahl der Unfälle pro Vekehrsmittel in Bayern.
def accident_vehicles(df_unfallatlas):
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
accident_vehicles(df_unfallatlas)


#Darstellung der Anzahl der Unfälle pro Bundesland.
def number_of_accidents_per_state(df_unfallatlas):
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
number_of_accidents_per_state(df_unfallatlas)


#Darstellung der Anzahl der Unfälle pro Unfallkategorie in Bayern.
def number_of_accidents_per_category(df_unfallatlas):
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
number_of_accidents_per_category(df_unfallatlas)


#Darstellung der Wochentage mit den meisten Unfällen pro Jahr in München.
def accidents_per_weekday_heatmap(df_unfallatlas):
    accidents_munich = df_unfallatlas.loc[df_unfallatlas['AGS'] == 9162000]

    weekdays = ['Sonntag','Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag']
    accweekday = accidents_munich.groupby(['UJAHR', 'UWOCHENTAG']).size()
    accweekday = accweekday.rename_axis(['UJAHR', 'UWOCHENTAG']).unstack('UWOCHENTAG')
    accweekday.columns = weekdays
    accweekday = accweekday[['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag']]

    plt.figure(figsize=(15, 10))
    sns.heatmap(accweekday, cmap='plasma_r')
    plt.title('Anzahl der Unfälle nach Wochentagen pro Jahr in München', fontsize=25, pad=20)
    plt.ylabel('')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
accidents_per_weekday_heatmap(df_unfallatlas)


#Darstellung der Monate mit den meisten Unfällen pro Jahr in München.
def accidents_per_month_heatmap(df_unfallatlas):
    accidents_munich = df_unfallatlas.loc[df_unfallatlas['AGS'] == 9162000]

    weekdays = ['Sonntag','Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag']
    accweekday = accidents_munich.groupby(['UMONAT', 'UWOCHENTAG']).size()
    accweekday = accweekday.rename_axis(['UJAHR', 'UWOCHENTAG']).unstack('UWOCHENTAG')
    accweekday.columns = weekdays
    accweekday = accweekday[['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag']]

    plt.figure(figsize=(15, 10))
    sns.heatmap(accweekday, cmap='plasma_r')
    plt.title('Anzahl der Unfälle nach Wochentagen pro Monat in München', fontsize=25, pad=20)
    plt.ylabel('')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
accidents_per_month_heatmap(df_unfallatlas)


#Darstellung der Stunden mit den meisten Unfällen pro Jahr in München.
def accidents_per_hour_heatmap(df_unfallatlas):
    accidents_munich = df_unfallatlas.loc[df_unfallatlas['AGS'] == 9162000]

    weekdays = ['Sonntag','Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag']
    accweekday = accidents_munich.groupby(['USTUNDE', 'UWOCHENTAG']).size()
    accweekday = accweekday.rename_axis(['UJAHR', 'UWOCHENTAG']).unstack('UWOCHENTAG')
    accweekday.columns = weekdays
    accweekday = accweekday[['Montag', 'Dienstag', 'Mittwoch', 'Donnerstag', 'Freitag', 'Samstag', 'Sonntag']]

    plt.figure(figsize=(15, 10))
    sns.heatmap(accweekday, cmap='plasma_r')
    plt.title('Anzahl der Unfälle nach Wochentagen pro Stunde in München', fontsize=25, pad=20)
    plt.ylabel('')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.show()
accidents_per_hour_heatmap(df_unfallatlas)


#Darstellung der Verteilung der jeweiligen Attribute, bezogen auf das Bundesland Bayern.
def feature_distribution(df_unfallatlas):
    df_unfallatlas_munich = df_unfallatlas.loc[(df_unfallatlas['ULAND'] == 9)]
    df_unfallatlas_munich['UWOCHENTAG'] = df_unfallatlas['UWOCHENTAG'].replace({1:7, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6})
    fig, axes = plt.subplots(2, 4, figsize=(15, 10))
    fig.suptitle('Merkmalsverteilung der Attribute hinsichtlich der Unfälle für das Bundesland Bayern', fontsize = 25)

    p1=sns.countplot(df_unfallatlas_munich['UJAHR'] , ax=axes[0,0])
    p2=sns.countplot(df_unfallatlas_munich['UMONAT'],  ax=axes[0,1])
    p3=sns.countplot(df_unfallatlas_munich['UWOCHENTAG'],  ax=axes[0,2])
    p4=sns.countplot(df_unfallatlas_munich['USTUNDE'], ax=axes[0,3])
    p5=sns.countplot(df_unfallatlas_munich['UART'], ax=axes[1,0])
    p6=sns.countplot(df_unfallatlas_munich['UTYP1'], ax=axes[1,1])
    p7=sns.countplot(df_unfallatlas_munich['STRZUSTAND'],  ax=axes[1,2])
    p8=sns.countplot(df_unfallatlas_munich['ULICHTVERH'],  ax=axes[1,3])

    p1.set(ylabel=None)
    p1.tick_params(labelsize=10)
    p1.set_xlabel('Jahr',fontsize = 15)
    p2.set(ylabel=None)
    p2.tick_params(labelsize=10)
    p2.set_xlabel('Monat',fontsize = 15)
    p3.set(ylabel=None)
    p3.tick_params(labelsize=10)
    p3.set_xlabel('Wochentag',fontsize = 15)
    p4.set(ylabel=None)
    p4.tick_params(labelsize=9)
    p4.set_xlabel('Stunde',fontsize = 15)
    p5.set(ylabel=None)
    p5.tick_params(labelsize=9)
    p5.set_xlabel('Unfallart',fontsize = 15)
    p6.set(ylabel=None)
    p6.tick_params(labelsize=10)
    p6.set_xlabel('Unfalltyp',fontsize = 15)
    p7.set(ylabel=None)
    p7.tick_params(labelsize=10)
    p7.set_xlabel('Straßenzustand',fontsize = 15)
    p8.set(ylabel=None)
    p8.tick_params(labelsize=10)
    p8.set_xlabel('Lichtverhältnisse',fontsize = 15)

    plt.tight_layout()
    plt.show()
feature_distribution(df_unfallatlas)


#Korrelation Alkoholkonsum und Unfallanzahl bezogen auf den Unfalldaten für Deutschland.
def correlation_alcohol_and_car_accidents(df_unfallatlas):
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

#correlation_alcohol_and_car_accidents(df_unfallatlas)
