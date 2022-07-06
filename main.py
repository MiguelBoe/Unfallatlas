import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import joblib
import folium
from folium import plugins
import webbrowser
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from data_preprocessing import get_data, preprocessing, pred_IstGkfz, get_wheater_data, prepare_number_of_accidents, add_exog_data
from utils import *
from models import *

'''
Zunächst werden in dem Konfigurationsabschnitt die wesentlichen Parameter für das Tool festgelegt.
'''
#Konfiguration
selection = None
ags = '09162000' #München
model_features = ['Temperatur Mittelwert', 'Niederschlagmenge in Summe Liter pro qm', 'Sonnenscheindauer in Summe in Stunden'] #Für das SARIMAX-Modell
visualization_mode = False #Bei True werden zusätzliche Plots generiert
model_accident_severity = 'random_forest_model' #decision_tree_model, random_forest_model, gaussian_nb_model, svm_model, knn_model
undersampling_mode = 'random' #random, nearmiss, False

'''
Einlesen der exogenen Daten (Wetterdaten). Benutzt wurden lediglich die Wetterdaten der Stadt München, für die Vorhersage
der Unfallzahlen. Die Wetterdaten, welche sich auf ganz Deutschland beziehen, haben nicht bei der Vorhersage der Unfall-
kategorie geholfen.
'''
#Exogene Daten
wheater_data_munich = pd.read_csv('data/exog_data/Wetterdaten_München.csv', sep =';')
#wheater_data_ger = pd.read_csv('exog_data/Wetterdaten_Deutschland.csv', sep = ';')

#Begrüßung.
print('\n##################################################')
print('Willkommen beim Unfallvorhersage-Tool für München!')
print('##################################################\n')
print('Einlesen und verarbeiten der Daten ...')

'''
Im nächsten Abschnitt werden die Unfalldaten eingelesen. Dabei wird mit Exception Handling geprüft, ob bereits eine auf-
bereitete CSV-Datei (df_unfallatlas.csv) mit den Daten vorhanden ist. Ist dies der Fall, wird diese Datei einfach eingelesen.
Wenn nicht, werden die einzelnen CSV-Dateien mit den Rohdaten eingelesen und dann mit verschiedenen Funktionen aufbereitet. 
Diese Funktionen befinden sich in der Datei data_preprocessing und werden dort beschrieben.
'''
# Einlesen und verarbeiten der Unfalldaten.
try:
    df_unfallatlas = pd.read_csv('data/df_unfallatlas.csv', delimiter=';', low_memory=False)
    df_unfallatlas['AGS'] = df_unfallatlas['AGS'].astype(str).str.zfill(8)
except:
    df_unfallatlas = get_data()
    df_unfallatlas = preprocessing(df_unfallatlas)
    df_unfallatlas = pred_IstGkfz(df_unfallatlas)
    df_unfallatlas.to_csv('data/df_unfallatlas.csv', sep=  ';', index=False)

'''
Als nächstes wird die Funktion get_wheater_data() aufgerufen, welche in der Datei data_preprocessing.py abgespeichert ist und
die exogenen Daten (Wetterdaten) aufbereitet. Da die Wetterdaten für ganz Deutschland nicht bei der Vorhersage der Unfallkategorie 
geholfen haben, wurde die Funktion zur Aufbereitung der Wetterdaten deaktiviert und nicht weiterentwickelt.
'''
#Einlesen und verarbeiten der Wetterdaten.
wheater_data_munich = get_wheater_data(wheater_data_munich)
#wheater_data_ger = get_wheater_data(wheater_data_ger)

#Hinzufügen der exogenen Daten.
#df_unfallatlas = add_exog_data(df_unfallatlas, wheater_data_ger)
print('Daten verarbeitet!')

'''
Beginn der interaktiven Anwendung. Als erstes wird der Nutzer danach gefragt, was er tun will. Er kann dabei mit einer binären
Entscheidung reagieren. Bei 0 wählt er die Vorhersage der schwere eines Unfalls und bei 1 die Vorhersage der Anzahl der Unfälle 
für das Jahr 2021. Damit der Nutzer keine falschen Eingaben tätigen kann, werden mit Hilfe der Funktion query_exception()
fehlerhafte Eingaben abgefangen. Diese Funktion ist in der Datei utils.py abgespeichert.
'''
#Auswahl der Vorhersage.
message = '\nWas wollen Sie tun?'
print(message, '\n')
tool = query_exception(dict = tools, message = message)

if tool == 0:
    print('\n#####################################')
    print('Vorhersage der schwere eines Unfalls.')
    print('#####################################\n')
    print('Erstellung des Modells ...\n')

    '''
    In der Funktion baseline_model wird die Baseline Methode validiert. Da dies lediglich einmal erfolgen musste, wurde
    diese Funktion deaktiviert. Die Funktion befindet sich in der models.py-Datei, in welcher diese genauer erklärt wird.
    '''
    #Validierung des Baseline Modells.
    #baseline_clf_report = baseline_model(df_unfallatlas)

    '''
    Im Abschnitt "Laden des Modells" wird mit Exception Handling versucht ein bereits erlerntes Modell einzulesen, um dieses
    für die Vorhersage der Unfallkategorie zu nutzen. Das erneute Training eines Modells kann nämlich etwas Zeit in Anspruch
    nehmen. Wenn jedoch noch kein Modell trainiert und abgespeichert wurde, wird mit Hilfe einer der untenstehenden Funktionen
    ein Modell trainiert. Beziehungsweise das Modell, welches oben im Konfigurationsbereich bestimmt wurde. Nach dem Training 
    wird das Modell dann als SAV-Datei abgespeichert, damit bei der erneuten Ausfühtung der Vorhersage, das Tool nicht erneut
    trainiert werden muss.
    '''
    #Laden des Modells.
    try:
        model = joblib.load(f'models/{model_accident_severity}.sav')
    except:
        if model_accident_severity == 'knn_model':
            model = pred_accident_severity_nearest_neighbors(df_unfallatlas, undersampling_mode)
        elif model_accident_severity == 'decision_tree_model':
            model = pred_accident_severity_decision_tree(df_unfallatlas, undersampling_mode)
        elif model_accident_severity == 'random_forest_model':
            model = pred_accident_severity_random_forest(df_unfallatlas, undersampling_mode)
        elif model_accident_severity == 'gaussian_nb_model':
            model = pred_accident_severity_gaussian_nb(df_unfallatlas, undersampling_mode)
        elif model_accident_severity == 'svm_model':
            model = pred_accident_severity_svm(df_unfallatlas, undersampling_mode)

    '''
    Im Abschnitt "Abfrage der Unfalldaten" wird die Funktion query() ausgeführt, mit welcher die Unfallinformationen abgefragt
    werden. Diese Funktion ist in der Datei utils.py abgespeichert und wird dort erklärt. 
    '''
    #Abfrage der Unfalldaten.
    prediction = query()

    '''
    Nachfolgend werden anhand eines statistischen Verfahrens basierend auf der Abfrage und den historischen Unfalldaten 
    die Wahrscheinlichkeiten der Unfallkategorien ermittelt. Diese Wahrscheinlichkeiten werden allerdings nur berechnet, 
    wenn es Datenpunkte gibt, welche exakt den Abfrageparametern entsprechen. Ist dies der Fall, werden die Wahrscheinlich-
    keiten der Unfallkategorien zu der Abfrage ausgegeben. Wenn dies nicht der Fall ist, wird lediglich die Unfallkategorie 
    anhand des Modells und der Abfrageparameter prognostiziert und ausgegeben. Die Unfallkategorie wird in beiden Fällen
    anhand des ML-Modells prognostiziert. 
    '''
    try:
        #Bestimmung der Unfallkategorie.
        accident_severity = statistical_determination_accident_severity(df_unfallatlas, prediction)

        #Berechnung der Wahrscheinlichkeit der Unfallkategorie.
        accident_severity_probability = round(accident_severity.groupby('UKATEGORIE').size().div(len(accident_severity))*100, 2)

        #Bestimmung der Unfallkategorie.
        accident_severity = model.predict(prediction)

        #Ausgabe der Wahrscheinlichkeit der Unfallkategorien.
        if len(accident_severity_probability) != 0:
            print('\nHistorische Wahrscheinlichkeit der Unfallkategorie:')

        for probability in accident_severity_probability:
            print(f'{kategorien[accident_severity_probability[accident_severity_probability == probability].index[0]]}\t', probability, '%')

        #Ausgabe der vorhergesagten Unfallkategorie.
        print('\nUnfallkategorie:\t', kategorien[accident_severity[0]])
        print('###################################################\n')

    except:
        #Bestimmung der Unfallkategorie.
        accident_severity = model.predict(prediction)

        #Ausgabe der Unfallkategorie
        print('\nUnfallkategorie:\t', kategorien[accident_severity[0]])
        print('################################################\n')

elif tool == 1:
    print('\n####################################################')
    print('Vorhersage der Anzahl der Unfälle für das Jahr 2021.')
    print('####################################################\n')

    '''
    Bei der Vorhersage der Unfallzahlen wird zunächst anhand des aufbereiteten Datensatzes und des AGS sowie der exogenen 
    Wetterdaten eine Zeitreihe erstellt. Die Unfallzahlen werden dabei auf Monatsbasis aggregiert. Dies geschieht in der 
    Funktion prepare_number_of_accidents(). Anschließend wird in der Funktion grid_search() mit Hilfe eines GridSearch-
    Algorithmus und Cross Validation die beste Hyperparameter-Kombination für das SARIMAX-Modell gesucht. Zuletzt wird 
    mit den gefundenen Parametern das SARIMAX-Modell trainiert.
    '''
    #Erstellung des Modells mit vorheriger Grid Search zur Definition der besten Parameter für das Modell.
    df_number_of_accidents, wheater_data_munich_2021 = prepare_number_of_accidents(df_unfallatlas, ags, wheater_data_munich, visualization_mode)
    bestAIC, bestParam, bestSParam = grid_search(y = df_number_of_accidents['Number of Accidents'], x = df_number_of_accidents[model_features])
    sarima = sarima(bestParam, bestSParam, visualization_mode, y = df_number_of_accidents['Number of Accidents'], x = df_number_of_accidents[model_features])

    '''
    Nach dem Trainig des Modells werden anhand der Zeitreihe und mit Hilfe des Modells die Unfallzahlen für das Jahr 2021 berechnet.
    '''
    #Vorhersage der Anzahl der Unfälle für das Jahr 2021.
    pred_start, pred_end = str(np.min(df_number_of_accidents.index) + relativedelta(months = 48)), str(np.max(df_number_of_accidents.index) + relativedelta(months=12))
    prediction = sarima.get_prediction(pred_start, pred_end, exog=wheater_data_munich_2021[model_features])
    df_number_of_accidents = pd.concat([df_number_of_accidents['Number of Accidents'], round(prediction.predicted_mean.last('12M'))])
    df_number_of_accidents = pd.DataFrame(df_number_of_accidents).rename(columns = {0: 'Number of Accidents'})

    '''
    Im nachfolgenden Abschnitt wurde das Modell validiert. Die Validierung erfolgte anhand der Maße MSE, MAE und MAPE. Dabei
    wurde überprüft, ob die Hinzunahme der exogenen Daten (Wetterdaten) das Ergebnis verbessert. Dies ist der Fall. Für die
    Validierung wurde die Länge des Training-Sets verkürzt, sodass das Modell mit unbekannten Daten validiert wurde. Anschließend
    wird die Prognose der Unfallzahlen auf Monatsbasis aggregiert für das Jahr 2021 ausgegeben. Zudem wird die Zeitreihe,
    inklusive der Prognose, in einem Plot visualisiert.
    '''
    #Validierung des SARIMAX Modells anhand des MSE und MAE bezogen auf das Jahr 2020. Vergleich der Scores mit dem Modell ohne exogene Variablen.
    mse_value = round(mean_squared_error(df_number_of_accidents['Number of Accidents'].loc[df_number_of_accidents.index.year == 2020], prediction.predicted_mean.head(12)), 2)
    mae_value = round(mean_absolute_error(df_number_of_accidents['Number of Accidents'].loc[df_number_of_accidents.index.year == 2020], prediction.predicted_mean.head(12)), 2)
    mape_value = round(mean_absolute_percentage_error(df_number_of_accidents['Number of Accidents'].loc[df_number_of_accidents.index.year == 2020], prediction.predicted_mean.head(12)), 2)

    #Ausgabe der Unfallzahlen pro Monat, bezogen auf das Jahr 2021.
    print('\nVorhersage der Unfallzahlen auf Monatsbasis:')
    print(round(prediction.predicted_mean.last('12M')))

    #Visualisierung der Zeitreihe inklusive der Prognose der Unfallzahlen.
    visualization_ts(df_number_of_accidents, prediction)

    print('\nSie können sich nun eine Map anzeigen lassen, in welcher die geährlichsten Unfallstellen in München dargstellt sind.')

    message = 'Die gefärhrlichsten Unfallstellen können sie für einen bestimmten Monat betrachten oder für das ganze Jahr:'
    print(message, '\n')
    period_map = query_exception(dict = monate_map, message = message)

    '''
    Anschließend können sich die historischen Unfallorte anhand der Koordinaten mit einer Heatmap auf OpenStreetMap angezeigt
    werden lassen. Vorab kann entschieden werden, für welchen Monat die Unfallorte angezeigt werden sollen. Es können aber 
    auch alle historischen Unfallorte angezeigt werden. Da die Unfallorte der Kategorien 1 und 2 am relevantesten sind, werden 
    nur diese auf der Karte dargestellt.
    '''
    #Vorbereitung des Datensatzes zur Darstellung der Unfallorte.
    if period_map == 0:
        df_unfallatlas_visualization = df_unfallatlas[(df_unfallatlas['AGS'] == ags) & (df_unfallatlas['UKATEGORIE'] != 3)].reset_index(drop=True)
    else:
        df_unfallatlas_visualization = df_unfallatlas[(df_unfallatlas['AGS'] == ags) &
                                                      (df_unfallatlas['UKATEGORIE'] != 3) &
                                                      (df_unfallatlas['UMONAT'] == period_map)].reset_index(drop=True)

    #Darstellung der Unfallorte.
    class Map:

        #init.
        def __init__(self, center, zoom_start, locationlist, locations):
            self.center = center
            self.zoom_start = zoom_start
            self.locationlist = locationlist

        def showMap(self):

            #Erstellung der Map.
            accident_map = folium.Map(location = self.center, zoom_start = self.zoom_start)
            if period_map != 0:
                for point in range(0, len(locationlist)):
                    folium.CircleMarker(locationlist[point], popup = arten[df_unfallatlas_visualization['UART'][point]], radius = 3).add_to(accident_map)
            accident_map.add_child(plugins.HeatMap(locations, radius = 15))

            #Darstellung der Map.
            accident_map.save('accident_map.html')
            webbrowser.open('accident_map.html')

    #Definition der Koordinaten.
    locations = df_unfallatlas_visualization[['lat', 'lon']]
    locationlist = locations.values.tolist()
    coords = [df_unfallatlas_visualization['lat'].mean(), df_unfallatlas_visualization['lon'].mean()]

    map = Map(center = coords, zoom_start = 12, locationlist = locationlist, locations = locations.values)
    map.showMap()

    print('\n####################################################################################################')
    print(f'In der Map sind die gefährlichsten Unfallorte in München für den Zeitraum: "{monate_map[period_map]}" dargestellt.')
    print('####################################################################################################\n')