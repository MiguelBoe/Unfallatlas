import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import researchpy
from data_preprocessing import get_data, preprocessing, pred_IstGkfz

#Begrüßung.
print('\nWillkommen beim Data-Exploration Bereich des Accident Prediction Tools!')
print('\nEinlesen und Verarbeiten der Daten ...')

#Einlesen und Verarbeiten der Daten.
df_unfallatlas = get_data()
df_unfallatlas = preprocessing(df_unfallatlas)
df_unfallatlas = pred_IstGkfz(df_unfallatlas)

print('\nAnalyse der Daten ...')

#Analyse der Daten______________________________________________________________________________________________________

#Darstellung der Verteilung der jeweiligen Attribute
sns.set_style('whitegrid')
sns.set(rc={"figure.figsize": (8, 4)})
verteilung = sns.distplot('UKATEGORIE')
plt.show()

#Darstellung der Korrelation der Attribute.
sns.set()
plt.figure(figsize=(15, 10))
plt.title('Korrelation der Attribute', fontsize=25, pad=20)
sns.heatmap(df_unfallatlas.corr(), annot=True, robust=True)
plt.show()

#Darstellung der Außreißer der Attribute
box_plot_1 = df_unfallatlas[['UKATEGORIE', 'UART', 'UTYP1', 'ULICHTVERH', 'STRZUSTAND', 'IstRad', 'IstPKW']].plot(kind='box', title='boxplot')
box_plot_2 = df_unfallatlas[['IstFuss', 'IstKrad', 'IstGkfz', 'IstSonstige',]].plot(kind='box', title='boxplot')
plt.show()

#Üperprüfung der Abhängikeut der Daten zu einer bestimment Variable
ch_sq_1= researchpy.crosstab(df_unfallatlas['UKATEGORIE'], df_unfallatlas['UART'], test= "chi-square", expected_freqs = True)
ch_sq_2= researchpy.crosstab(df_unfallatlas['UKATEGORIE'], df_unfallatlas['UTYP1'], test= "chi-square", expected_freqs= True)
ch_sq_3= researchpy.crosstab(df_unfallatlas['UKATEGORIE'], df_unfallatlas['STRZUSTAND'], test= "chi-square", expected_freqs=True)
ch_sq_4= researchpy.crosstab(df_unfallatlas['UKATEGORIE'], df_unfallatlas['IstRad'], test= "chi-square", expected_freqs=True)
ch_sq_5= researchpy.crosstab(df_unfallatlas['UKATEGORIE'], df_unfallatlas['IstPKW'], test= "chi-square", expected_freqs=True)
ch_sq_6= researchpy.crosstab(df_unfallatlas['UKATEGORIE'], df_unfallatlas['IstFuss'], test= "chi-square", expected_freqs=True)
ch_sq_7= researchpy.crosstab(df_unfallatlas['UKATEGORIE'], df_unfallatlas['IstKrad'], test= "chi-square", expected_freqs=True)
ch_sq_8= researchpy.crosstab(df_unfallatlas['UKATEGORIE'], df_unfallatlas['IstGkfz'], test= "chi-square", expected_freqs=True)
ch_sq_9= researchpy.crosstab(df_unfallatlas['UKATEGORIE'], df_unfallatlas['IstSonstige'], test= "chi-square", expected_freqs=True)
ch_sq_10= researchpy.crosstab(df_unfallatlas['UKATEGORIE'], df_unfallatlas['lon'], test= "chi-square", expected_freqs=True)
ch_sq_11= researchpy.crosstab(df_unfallatlas['UKATEGORIE'], df_unfallatlas['lan'], test= "chi-square", expected_freqs=True)

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