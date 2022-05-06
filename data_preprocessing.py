import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score


# Einlesen der Daten ___________________________________________________________________________________________________

def get_data():

    # Einlesen der Daten und Verknüpfung in einem DataFrame.
    files = glob.glob('data/*.csv')
    df_unfallatlas = pd.DataFrame()
    for f in files:
        csv = pd.read_csv(f, delimiter = ';', low_memory = False)
        df_unfallatlas = pd.concat([df_unfallatlas, csv])

    # Informationen über den Datensatz.
    # df_unfallatlas.info()
    # df_description = df_unfallatlas.describe()

    return df_unfallatlas

df_unfallatlas = get_data()


# Aufbereitung der Daten._______________________________________________________________________________________________

def preprocessing(df_unfallatlas):

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

    return df_unfallatlas

df_unfallatlas = preprocessing(df_unfallatlas = df_unfallatlas)

# Prediction IstGkfz.___________________________________________________________________________________________________

def pred_IstGkfz(df_unfallatlas):

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
    #print('\nAccuracy-Score des Modells:', round(score, 2))

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
    df_unfallatlas.loc[(df_unfallatlas['UJAHR'] == 2017) & (df_unfallatlas['IstSonstige'] == 1) & (df_unfallatlas['IstGkfz'] == 1), 'IstSonstige'] = 0

    return df_unfallatlas

df_unfallatlas = pred_IstGkfz(df_unfallatlas = df_unfallatlas)