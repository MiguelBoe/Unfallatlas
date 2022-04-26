import pandas as pd
import glob
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import folium
from folium import plugins
import webbrowser

# Einlesen der Daten und Verknüpfung in einem DataFrame._____________________________________________________________


# Einlesen der Daten.
files = glob.glob('data/*.csv')
df_unfallatlas = pd.DataFrame()
for f in files:
    csv = pd.read_csv(f, delimiter=';', low_memory=False)
    df_unfallatlas = pd.concat([df_unfallatlas, csv])

# Informationen über den Datensatz.
# df_unfallatlas.info()
# df_description = df_unfallatlas.describe()


# Aufbereitung der Daten._______________________________________________________________________________________________________


# Aktualisierung des Index.
df_unfallatlas = df_unfallatlas.reset_index(drop=True)

# Zusammenführen der Spalten.
df_unfallatlas['ULICHTVERH'] = df_unfallatlas['ULICHTVERH'].fillna(df_unfallatlas['LICHT'])
df_unfallatlas['STRZUSTAND'] = df_unfallatlas['STRZUSTAND'].fillna(df_unfallatlas['IstStrasse'])
df_unfallatlas['IstSonstige'] = df_unfallatlas['IstSonstige'].fillna(df_unfallatlas['IstSonstig'])

# Entfernen der überflüssigen Attribute.
df_unfallatlas.drop(['FID', 'OBJECTID', 'OBJECTID_1', 'LICHT', 'IstStrasse', 'IstSonstig',
                     'UIDENTSTLA', 'UIDENTSTLAE', 'LINREFX', 'LINREFY'], axis=1, inplace=True)

# Erstellung des AGS.
df_unfallatlas['AGS'] = df_unfallatlas['ULAND'].astype(str).str.zfill(2) \
                        + df_unfallatlas['UREGBEZ'].astype(str).str.zfill(1) \
                        + df_unfallatlas['UKREIS'].astype(str).str.zfill(2) \
                        + df_unfallatlas['UGEMEINDE'].astype(str).str.zfill(3)

# Umbenennung der Spalten YGCSWGS84 und XGCSWGS84.
df_unfallatlas.rename(columns={"YGCSWGS84": "lat"}, inplace=True)
df_unfallatlas.rename(columns={"XGCSWGS84": "lon"}, inplace=True)

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

# Prediction IstGkfz._____________________________________________________________________________________________________________________


# Erstellung eines Datensatzes zur Abschätzung der fehlenden IstGkfz für 2017.
df_unfallatlas.loc[df_unfallatlas.IstSonstige == 0, 'IstGkfz'] = 0
df_unfallatlas_istGkfz = df_unfallatlas[df_unfallatlas['IstGkfz'].notnull()]

# Definition von X und y.
X = df_unfallatlas_istGkfz.drop(['IstGkfz'], axis=1)
y = df_unfallatlas_istGkfz['IstGkfz']

# Splitten der Daten in Test- und Training-Set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Training der Daten.
decision_tree_classification = DecisionTreeClassifier().fit(X_train, y_train)

# Validierung des Modells.
results = pd.DataFrame(decision_tree_classification.predict(X_test), index=X_test.index)

# Überprüfung der Genauigkeit des Modells.
score = accuracy_score(y_test, results)
print('\nAccuracy-Score des Modells:', round(score, 2))

# Validierung der Predictions in einem DataFrame
results['IstGkfz'] = df_unfallatlas['IstGkfz']
results.rename(columns={0: "pred"}, inplace=True)
results.rename(columns={"IstGkfz": "test"}, inplace=True)

# Abschätzung der fehlenden IstGkfz-Werte für das Jahr 2017 mit dem Decision Tree Klassifikationsmodell.
df_unfallatlas_pred_gkfz = df_unfallatlas[df_unfallatlas['IstGkfz'].isnull()]
df_unfallatlas_pred_gkfz = df_unfallatlas_pred_gkfz.drop('IstGkfz', axis=1)
df_IstGkfz_2017 = pd.DataFrame(decision_tree_classification.predict(df_unfallatlas_pred_gkfz),
                               index=df_unfallatlas_pred_gkfz.index)

# Ergänzung der fehlenden IstGkfz-Werte im DataFrame df_unfallatlas.
df_unfallatlas['IstGkfz'] = df_unfallatlas['IstGkfz'].fillna(df_IstGkfz_2017[0])
df_unfallatlas.loc[df_unfallatlas.IstSonstige == 0, 'IstGkfz'] = 0
df_unfallatlas.loc[(df_unfallatlas['UJAHR'] == 2017) & (df_unfallatlas['IstSonstige'] == 1) & (
            df_unfallatlas['IstGkfz'] == 1), 'IstSonstige'] = 0

# Analyse._____________________________________________________________________________________________________________________


# Abfrage.
ags = input('\nGemeinde: ')
# year = int(input('Jahr: '))
category = int(input('Kategorie: '))
# monat = int(input('Monat: '))

df_unfallatlas_query = df_unfallatlas[
    # (df_unfallatlas['UJAHR'] == year) &
    (df_unfallatlas['AGS'] == ags) &
    (df_unfallatlas['UKATEGORIE'] == category)
    # & (df_unfallatlas['UMONAT'] == category)
    ]

df_unfallatlas_query = df_unfallatlas_query.reset_index(drop=True)


# Darstellung der Unfallorte.
class Map:
    def __init__(self, center, zoom_start, locationlist, locations):
        self.center = center
        self.zoom_start = zoom_start
        self.locationlist = locationlist

    def showMap(self):
        # Create the map
        my_map = folium.Map(location=self.center, zoom_start=self.zoom_start)
        for point in range(0, len(locationlist)):
            folium.CircleMarker(locationlist[point], popup=df_unfallatlas_query['UART'][point], radius=3).add_to(my_map)
        my_map.add_child(plugins.HeatMap(locations, radius=15))
        # Display the map
        my_map.save("map.html")
        webbrowser.open("map.html")


# Define coordinates of where we want to center our map
locations = df_unfallatlas_query[['lat', 'lon']]
locationlist = locations.values.tolist()
coords = [df_unfallatlas_query['lat'].mean(), df_unfallatlas_query['lon'].mean()]

map = Map(center=coords, zoom_start=12, locationlist=locationlist, locations=locations.values)
map.showMap()

# 05570020
# 05970040
# 05315000


# Predict Unfalltyp._____________________________________________________________________________________________________________________


# Darstellung der Korrelation der Attribute
plt.figure(figsize=(15, 10))
plt.title('Korrelation der Attribute', fontsize=25, pad=20)
sns.heatmap(df_unfallatlas.corr(), annot=True, robust=True)
plt.show()

# Definition von X und y.
X = df_unfallatlas.drop(['UTYP1', 'lat', 'lon'], axis=1)
y = df_unfallatlas['UTYP1']

# Splitten der Daten in Test- und Training-Set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Training der Daten.
decision_tree_classification = DecisionTreeClassifier(max_depth=10, random_state=1).fit(X_train, y_train)

# Validierung des Modells.
results = pd.DataFrame(decision_tree_classification.predict(X_test), index=X_test.index)

# Überprüfung der Genauigkeit des Modells.
score = accuracy_score(y_test, results)
print('\nAccuracy-Score des Modells:', round(score, 2))