import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
from pylab import rcParams
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import matplotlib
#from main import wheater_data_munich, wheater_data_ger

# Einlesen der Daten.___________________________________________________________________________________________________

'''
Einlesen der Rohdaten, bzw. der einzelnen Datensätze der Unfalldaten und Zusammenfügen zu einem großen DataFrame.
'''
def get_data():

    # Einlesen der Daten und Verknüpfung in einem DataFrame.
    files = glob.glob('data/raw_data/*.csv')
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

'''
Aufbereitung des Datensatzes: 
- Anpassung der Indizes. 
- Zusammenfügen der Spalten, die gleiche Inhalte haben, aber aufgrund verschiedener Bezeichnungen in den einzelnen Jahren getrennt wurden. 
- Entfernung von Attributen ohne einen nennenswerten Nutzen. 
- Erzeugung einer neuen Spalte mit dem AGS, welcher aus den zugehörigen Daten zusammengesetzt wurde. 
- Umbenennung der Koordinaten-Spalten und Formatierung ins amerikanische Format (Kommata zu Punkten). 
- Sortierung der Spalten.
'''
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

'''
In em Datensatz waren nun nur noch fehlende Werte in der Spalte IstGkfz enthalten. Diese fehlenden Werte sollten ergänzt
werden. Um dies zu erreichen, wurde ein Decision Tree-Klassifikationsmodell trainiert, welches anhand der restlichen Attribute
prognostiziert, ob an dem Unfall ein Gkfz beteiligt war oder nicht. Grund für das Fehlen der Werte ist, dass die Information
ob ein Gkfz an den Unfall beteiligt war oder nicht in manchen Jahren in der Spalte IstSonstige erfasst wurde. Somit wurden 
zunächst in allen Zeilen, in welchen in der Spalte IstSonstige eine Null steht, auch bei IstGkfz eine Null ergänzt, da hier 
mit 100 prozentiger Sicherheit kein Gkfz am Unfall beteiligt war. Somit mussten nur noch die Zeilen prognostiziert werden,
ich welchen in der Spalte IstSonstige eine 1 stand. Dies wurde gemacht und somit wurden alle fehlenden Werte in dem Datensatz
ergänzt. Die Genauigkeit der Methode war sehr gut, was bei der Validierung des Modells festgestellt werden konnte. Der Accuracy
Score lag bei über 99 %. Nach der erfolgreichen Validierung des Modells wurden die fehlenden IstGkfz-Werte im Datensatz 
prognostiziert und ersetzt. Nun war das Problem, dass in manchen Zeilen bei IstGkfz und IstSonstige eine 1 erfasst wurde. 
Dabei kann es einerseits sein, dass sowohl ein Güterkraftfahrzeug, als auch ein sonstiges Fahrzeug an dem Unfall beteiligt war. 
Andererseits kann es jedoch auch sein, dass ein Güterkraftfahrzeug hier doppelt erfasst wurde. Um letzteres auszuschließen, 
wurde in allen Zeilen, in denen bei IstGkfz und bei IstSonstige eine 1 erfasst wurde, der Eintrag in der Spalte IstSonstige 
auf 0 gesetzt. Dadurch kam es jedoch zu einer Verfälschung der Daten an den Stellen, bei denen sowohl ein Güterkraftfahrzeug, 
als auch ein Fahrzeug sonstiger Art beteiligt war, was bei dieser Methode jedoch in Kauf genommen werden musste. 
'''
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
    df_IstGkfz_2017 = pd.DataFrame(decision_tree_classification.predict(df_unfallatlas_pred_gkfz), index=df_unfallatlas_pred_gkfz.index)

    # Ergänzung der fehlenden IstGkfz-Werte im DataFrame df_unfallatlas.
    df_unfallatlas['IstGkfz'] = df_unfallatlas['IstGkfz'].fillna(df_IstGkfz_2017[0])
    df_unfallatlas.loc[(df_unfallatlas['UJAHR'] == 2017) & (df_unfallatlas['IstSonstige'] == 1) & (df_unfallatlas['IstGkfz'] == 1), 'IstSonstige'] = 0
    df_unfallatlas[['ULICHTVERH', 'STRZUSTAND', 'IstGkfz', 'IstSonstige']] = df_unfallatlas[['ULICHTVERH', 'STRZUSTAND', 'IstGkfz', 'IstSonstige']].astype('int64')

    return df_unfallatlas

df_unfallatlas = pred_IstGkfz(df_unfallatlas = df_unfallatlas)

# Einlesen und Vorbereiten der exogenen Daten.__________________________________________________________________________

'''
In der Funktion get_wheater_data() wurden die Wetterdaten eingelesen und aufbereitet, sodass diese der Zeitreihe für die
Prognose der Unfallzahlen hinzugefügt werden konnten.
'''
def get_wheater_data(wheater_data):

    wheater_data['Temperatur Mittelwert'] = wheater_data['Temperatur Mittelwert'].apply(lambda a: a.replace(",", ".")).astype(float)
    wheater_data['Niederschlagmenge in Summe Liter pro qm'] = wheater_data['Niederschlagmenge in Summe Liter pro qm'].apply(lambda a: a.replace(",", ".")).astype(float)
    wheater_data['Sonnenscheindauer in Summe in Stunden'] = wheater_data['Sonnenscheindauer in Summe in Stunden'].apply(lambda a: a.replace(",", ".")).astype(float)
    wheater_data = wheater_data.rename(columns={'Jahr':'year', 'Monat': 'month'})
    wheater_data['day'] = 1
    wheater_data = pd.DataFrame(wheater_data.set_index(pd.to_datetime(wheater_data[['year', 'month', 'day']])))
    wheater_data.drop(['year', 'month', 'day'], axis=1, inplace=True)

    return wheater_data

# Vorbereitung der Zeireihe und kurze Analyse.__________________________________________________________________________

'''
In der Funktion prepare_number_of_accidents() wird der DataFrame df_unfallatlas zu einer Zeitreihe formatiert, welche für
die Prognose der Unfallzahlen genutzt werden konnte. Zunächst wurden mit Hilfe des AGS lediglich die Unfalldaten für die Stadt 
München herausgefiltert. Danach wurde eine Spalte mit dem Datum des Unfallmonats angelegt und ins datetime-Format umge-
wandelt. Wenn im Konfigurationsbereich der visualization_mode aktiviert wurde, werden danach Plots dargestellt, welche be-
stimmte Eigenschaften der Zeitreihe beschreiben. Dies ist einmal die Dekomposition der Zeitreihe sowie die Darstellung der
Autocorrelation sowie der Partial Autocorrelation. Danach wurde die Zeitreihe dann mit dem ADFuller-Test auf Stationarität
überprüft. Dabei kam heraus, dass die Zeitreihe nicht stationär ist. Aus diesem Grund wurde sich dazu entschieden, ein 
SARIMAX-Modell für die Prognose der Unfallzahlen zu wählen, da dieses mit nicht stationären Zeitreihen umgehen kann.
Anschließend wurden in der untenstehenden Funktion der Zeitreihe die Wetterdaten für die Stadt München hinzugefügt.
'''
def prepare_number_of_accidents(df_unfallatlas, ags, wheater_data_munich, visualization_mode):

    #Index to datetime. Allerdings Problem wegen des Tages. Dieser ist ja nicht angegeben. Habe für Testzwecke mal den Wochentag genommen.
    df_number_of_accidents = df_unfallatlas[(df_unfallatlas['AGS'] == ags)].reset_index(drop = True)
    df_number_of_accidents.rename(columns={'UJAHR': 'year', 'UMONAT': 'month', 'UWOCHENTAG': 'day', 'UKATEGORIE': 'Number of Accidents'},inplace=True)
    df_number_of_accidents = pd.DataFrame(df_number_of_accidents.set_index(pd.to_datetime(df_number_of_accidents[['year', 'month', 'day']])).resample('M')['Number of Accidents'].count())
    df_number_of_accidents.index = df_number_of_accidents.index.map(lambda t: t.replace(day = 1))
    df_number_of_accidents.index.freq = 'MS'

    #Plot Dekomposition.
    if visualization_mode:
        rcParams['figure.figsize'] = 15, 10
        decomposition = sm.tsa.seasonal_decompose(df_number_of_accidents['Number of Accidents'], model = 'additive')
        fig = decomposition.plot()
        plt.show()

        #Plot ACF und PACF.
        plot_acf(df_number_of_accidents['Number of Accidents'])
        matplotlib.pyplot.show()
        plot_pacf(df_number_of_accidents['Number of Accidents'], method='ywm')
        matplotlib.pyplot.show()

    #Sind die Daten stationär?
    adf_test = adfuller(df_number_of_accidents, autolag='AIC')
    print('1. ADF : ', adf_test[0])
    print('2. P-Value : ', adf_test[1])
    print('3. Num Of Lags : ', adf_test[2])
    print('4. Num Of Observations Used For ADF Regression and Critical Values Calculation :', adf_test[3])
    print('5. Critical Values :')

    for key, val in adf_test[4].items():
        print('\t', key, ': ', val)

    df_number_of_accidents = pd.concat([df_number_of_accidents, wheater_data_munich], axis = 1)
    wheater_data_munich_2021 = df_number_of_accidents.tail(12)
    df_number_of_accidents = df_number_of_accidents.dropna()

    return df_number_of_accidents, wheater_data_munich_2021

'''
In der Funktion add_exog_data() werden dem DataFrame df_unfallatlas die Wetterdaten für ganz Deutschland auf Bundesland-
Ebene für die Prognose der Unfallkategorie hinzugefügt. Allerdings hat die Validierung gezeigt, dass die Genauigkeit des
Modells durch die Hinzunahme der Daten nicht verbessert werden konnte. Aus diesem Grund, wird die Funktion nicht weiter 
genutzt. Die Prognose der Unfallkategorie erfolgt also ohne exogene Daten.
'''
def add_exog_data(df_unfallatlas, wheater_data_ger):

    wheater_data_ger['UJAHR'] = wheater_data_ger.index.year
    wheater_data_ger['UMONAT'] = wheater_data_ger.index.month
    df_unfallatlas = df_unfallatlas.merge(wheater_data_ger, how='outer', on= ['ULAND', 'UJAHR', 'UMONAT'])
    df_unfallatlas.dropna(inplace=True)

    return df_unfallatlas