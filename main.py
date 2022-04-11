import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Einlesen der Daten und Verknüpfung in einem DataFrame.________________________________________________________________


# Einlesen der Daten.
files = glob.glob('data/*.csv')
df_unfallatlas = pd.DataFrame()
for f in files:
    csv = pd.read_csv(f, delimiter=';', low_memory = False)
    df_unfallatlas = pd.concat([df_unfallatlas, csv])

# Informationen über den Datensatz.
df_unfallatlas.info()
df_description = df_unfallatlas.describe()

# Aufbereitung der Daten._______________________________________________________________________________________________


# Aktualisierung des Index.
df_unfallatlas = df_unfallatlas.reset_index(drop = True)

# Zusammenführen der Spalten.
df_unfallatlas['ULICHTVERH'] = df_unfallatlas['ULICHTVERH'].fillna(df_unfallatlas['LICHT'])
df_unfallatlas['STRZUSTAND'] = df_unfallatlas['STRZUSTAND'].fillna(df_unfallatlas['IstStrasse'])
df_unfallatlas['IstSonstige'] = df_unfallatlas['IstSonstige'].fillna(df_unfallatlas['IstSonstig'])

# Entfernen der überflüssigen Attribute.
df_unfallatlas.drop(['FID', 'OBJECTID', 'OBJECTID_1', 'LICHT', 'IstStrasse', 'IstSonstig',
                     'UIDENTSTLA', 'UIDENTSTLAE', 'LINREFX', 'LINREFY'], axis = 1, inplace = True)

# Erstellung des AGS.
df_unfallatlas['AGS'] = df_unfallatlas['ULAND'].astype(str).str.zfill(2) \
                        + df_unfallatlas['UREGBEZ'].astype(str).str.zfill(1) \
                        + df_unfallatlas['UKREIS'].astype(str).str.zfill(2) \
                        + df_unfallatlas['UGEMEINDE'].astype(str).str.zfill(3)

# Umbenennung der Spalten YGCSWGS84 und XGCSWGS84.
df_unfallatlas.rename(columns = {"YGCSWGS84": "lat"}, inplace=True)
df_unfallatlas.rename(columns = {"XGCSWGS84": "lon"}, inplace=True)

# Konvertierung der Standortspalten von europäischen Komma-Sep-Werten in punktgetrennte Werte.
df_unfallatlas["lon"] = df_unfallatlas["lon"].apply(lambda a: a.replace(",", "."))
df_unfallatlas["lat"] = df_unfallatlas["lat"].apply(lambda a: a.replace(",", "."))

# Umformatierung der Standortspalten in Floats.
df_unfallatlas["lon"] = df_unfallatlas["lon"].astype(float)
df_unfallatlas["lat"] = df_unfallatlas["lat"].astype(float)

# Sortieren der Spalten.
df_unfallatlas.insert(0, 'AGS', df_unfallatlas.pop('AGS'))
df_unfallatlas.insert(21, 'lon', df_unfallatlas.pop('lon'))
df_unfallatlas.insert(21, 'lat', df_unfallatlas.pop('lat'))
df_unfallatlas.insert(13, 'STRZUSTAND', df_unfallatlas.pop('STRZUSTAND'))

# Prediction IstGkfz.___________________________________________________________________________________________________


# Darstellung fehlender Werte im Datensatz.
sns.set()
fig, ax = plt.subplots(figsize=(15, 10))
sns.heatmap(df_unfallatlas.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.title('Fehlende Werte im Datensatz (in gelb)', fontsize=30, pad=20)
plt.ylabel('Datenpunkte', fontsize=25)
plt.xlabel('AGS', fontsize=25)
ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.show()

# Erstellung eines Datensatzes zur Abschätzung der fehlenden IstGkfz für 2017.
df_unfallatlas_istGkfz = df_unfallatlas[df_unfallatlas['IstGkfz'].notnull()]

# Define x and y.
X = df_unfallatlas_istGkfz.drop(['IstGkfz'], axis=1)
y = df_unfallatlas_istGkfz['IstGkfz']

# Splitten der Daten in Test- und Training-Set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

# Training der Daten.
decision_tree_classification = DecisionTreeClassifier().fit(X_train, y_train)

# Validierung des Modells.
results = pd.DataFrame(decision_tree_classification.predict(X_test), index=X_test.index)

# Überprüfung der Genauigkeit des Modells.
score = accuracy_score(y_test, results)
print('\nAccuracy-Score des Modells:', round(score, 2))

# Validierung der Predictions in einem DataFrame
results['IstGkfz'] = df_unfallatlas['IstGkfz']

# Abschätzung der fehlenden IstGkfz-Werte für das Jahr 2017 mit dem Decision Tree Klassifikationsmodell.
df_unfallatlas_pred_gkfz = df_unfallatlas[df_unfallatlas['IstGkfz'].isnull()]
df_unfallatlas_pred_gkfz = df_unfallatlas_pred_gkfz.drop('IstGkfz', axis=1)
istGkfz_2017 = pd.DataFrame(decision_tree_classification.predict(df_unfallatlas_pred_gkfz),
                            index=df_unfallatlas_pred_gkfz.index)

# Ergänzung der fehlenden IstGkfz-Werte im DataFrame df_unfallatlas.
df_unfallatlas['IstGkfz'] = df_unfallatlas['IstGkfz'].fillna(istGkfz_2017[0])

# Analyse der Daten.____________________________________________________________________________________________________


# Ermittlung der Anzahl der Unfälle in den einzelnen Gemeinden in den Jahren von 2016 - 2020.
df_unfallatlas_count = df_unfallatlas.groupby('AGS').size().reset_index(name = 'number_of_accidents')
df_unfallatlas_count = df_unfallatlas_count.sort_values('number_of_accidents', ascending=False)

# Darstellung der 10 Gemeinden mit den meisten Unfällen in den Jahren von 2016 - 2020.
sns.set()
fig, ax = plt.subplots(figsize=(15, 10))
sns.barplot(x='AGS', y='number_of_accidents', data=df_unfallatlas_count.head(10))
plt.title('Die 10 Gemeinden mit den meisten Unfällen von 2016 - 2020', fontsize=30, pad=20)
plt.ylabel('Anzahl der Unfälle', fontsize=25)
plt.xlabel('AGS', fontsize=25)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=20)
plt.show()

# Darstellung der Korrelation der Attribute
plt.figure(figsize=(15, 10))
plt.title('Korrelation der Attribute', fontsize=25, pad=20)
sns.heatmap(df_unfallatlas.corr(), annot=True, robust=True)
plt.show()

# Abfragen._____________________________________________________________________________________________________________


# Ausgabe der Unfälle in Siegen im Jahr 2020.
df_unfallatlas_2020 = df_unfallatlas.loc[df_unfallatlas['UJAHR'] == 2020]
print('\nAnzahl der Unfälle in Siegen:', len(df_unfallatlas_2020.loc[df_unfallatlas_2020['AGS'] == '05970040']))

# Abfrage der Anzahl der Unfälle nach Jahr und AGS.
jahr = int(input('\nJahr: '))
ags = input('Gemeinde: ')
df_unfallatlas_year = df_unfallatlas.loc[df_unfallatlas['UJAHR'] == jahr]
print(f'\nAnzahl der Unfälle in {ags}:', len(df_unfallatlas_year.loc[df_unfallatlas_year['AGS'] == ags]))


