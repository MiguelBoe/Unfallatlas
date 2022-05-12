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
fig, axes = plt.subplots(2, 5, figsize=(18, 10))
fig.suptitle('Verteilung der jeweiligen Attribute')
Verteilung_1 = sns.distplot(df_unfallatlas['UKATEGORIE'], ax=axes[0,0])
Verteilung_1= sns.distplot(df_unfallatlas['UTYP1'], ax=axes[0,1])
Verteilung_1= sns.distplot(df_unfallatlas['ULICHTVERH'], ax=axes[0,2])
Verteilung_1 = sns.distplot(df_unfallatlas['STRZUSTAND'], ax=axes[0,3])
Verteilung_1= sns.distplot(df_unfallatlas['IstRad'], ax=axes[0,4])
Verteilung_1= sns.distplot(df_unfallatlas['IstPKW'], ax=axes[1,0])
Verteilung_1= sns.distplot(df_unfallatlas['IstFuss'], ax=axes[1,1])
Verteilung_2= sns.distplot(df_unfallatlas['IstKrad'], ax=axes[1,2])
Verteilung_2= sns.distplot(df_unfallatlas['IstGkfz'], ax=axes[1,3])
Verteilung_2= sns.distplot(df_unfallatlas['IstSonstige'], ax=axes[1,4])
plt.show()

#Darstellung eines Scatterplots
g = sns.scatterplot(data = df_unfallatlas['UKATEGORIE'])
g.set(xscale="log")

#Darstellung der Korrelation der Attribute.
sns.set()
plt.figure(figsize=(15, 10))
plt.title('Korrelation der Attribute', fontsize=25, pad=20)
corr = df_unfallatlas.corr(method = 'spearman')
sns.heatmap(corr, annot=True, robust=True)
plt.show()

#Darstellung der Außreißer der Attribute
box_plot_1 = df_unfallatlas[['UKATEGORIE', 'UTYP1', 'ULICHTVERH', 'STRZUSTAND', 'IstRad']].plot(kind='box', title='boxplot')
box_plot_2 = df_unfallatlas[['IstPKW','IstFuss', 'IstKrad', 'IstGkfz', 'IstSonstige',]].plot(kind='box', title='boxplot')
plt.show()

#Üperprüfung der Abhängikeut der Daten zu einer bestimment Variable
#ch_sq_1= researchpy.crosstab(df_unfallatlas['UKATEGORIE'], df_unfallatlas['UART'], test= "chi-square", expected_freqs = True)
#ch_sq_2= researchpy.crosstab(df_unfallatlas['UKATEGORIE'], df_unfallatlas['UTYP1'], test= "chi-square", expected_freqs= True)
#ch_sq_3= researchpy.crosstab(df_unfallatlas['UKATEGORIE'], df_unfallatlas['STRZUSTAND'], test= "chi-square", expected_freqs=True)
#ch_sq_4= researchpy.crosstab(df_unfallatlas['UKATEGORIE'], df_unfallatlas['IstRad'], test= "chi-square", expected_freqs=True)
#ch_sq_5= researchpy.crosstab(df_unfallatlas['UKATEGORIE'], df_unfallatlas['IstPKW'], test= "chi-square", expected_freqs=True)
#ch_sq_6= researchpy.crosstab(df_unfallatlas['UKATEGORIE'], df_unfallatlas['IstFuss'], test= "chi-square", expected_freqs=True)
#ch_sq_7= researchpy.crosstab(df_unfallatlas['UKATEGORIE'], df_unfallatlas['IstKrad'], test= "chi-square", expected_freqs=True)
#ch_sq_8= researchpy.crosstab(df_unfallatlas['UKATEGORIE'], df_unfallatlas['IstGkfz'], test= "chi-square", expected_freqs=True)
#ch_sq_9= researchpy.crosstab(df_unfallatlas['UKATEGORIE'], df_unfallatlas['IstSonstige'], test= "chi-square", expected_freqs=True)
#ch_sq_10= researchpy.crosstab(df_unfallatlas['UKATEGORIE'], df_unfallatlas['lon'], test= "chi-square", expected_freqs=True)
#ch_sq_11= researchpy.crosstab(df_unfallatlas['UKATEGORIE'], df_unfallatlas['lan'], test= "chi-square", expected_freqs=True)

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