import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from data_preprocessing import get_data, preprocessing, pred_IstGkfz, prepare_number_of_accidents

#Begrüßung.
print('\nWillkommen beim Data-Exploration Bereich des Accident Prediction Tools!')
print('\nEinlesen und Verarbeiten der Daten ...')

#Einlesen und Verarbeiten der Daten.
df_unfallatlas = get_data()
df_unfallatlas = preprocessing(df_unfallatlas)
df_unfallatlas = pred_IstGkfz(df_unfallatlas)

print('\nAnalyse der Daten ...')

#Analyse der Daten______________________________________________________________________________________________________

#Darstellung der Korrelation der Attribute.
sns.set()
plt.figure(figsize=(15, 10))
plt.title('Korrelation der Attribute', fontsize=25, pad=20)
sns.heatmap(df_unfallatlas.corr(), annot=True, robust=True)
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

number_of_accidents = pd.DataFrame(prepare_number_of_accidents(df_unfallatlas, ags = '09162000'))

print()