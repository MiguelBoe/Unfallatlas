import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import joblib
import folium
from folium import plugins
import webbrowser
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from data_preprocessing import get_data, preprocessing, pred_IstGkfz, get_wheater_data, prepare_number_of_accidents, add_exog_data
from utils import query_exception, query, kategorien, tools, monate_map, arten
from model import sarima, pred_accident_severity_decision_tree, pred_accident_severity_nearest_neighbors, \
                  grid_search, visualization_ts, pred_accident_severity_gaussian_nb

#Parameter
selection = None
ags = '09162000'
model_features = ['Temperatur Mittelwert', 'Niederschlagmenge in Summe Liter pro qm', 'Sonnenscheindauer in Summe in Stunden']
visualization_mode = False
load_model_accident_severity = 'knn_model' #decision_tree_model, knn_model, gaussian_nb_model

#Exogene Daten
wheater_data_munich = pd.read_csv('data/exog_data/Wetterdaten_München.csv', sep =';')
#wheater_data_ger = pd.read_csv('exog_data/Wetterdaten_Deutschland.csv', sep = ';')

#Begrüßung.
print('\n##################################################')
print('Willkommen beim Unfallvorhersage-Tool für München!')
print('##################################################\n')
print('Einlesen und verarbeiten der Daten ...')

# Einlesen und verarbeiten der Unfalldaten.
try:
    df_unfallatlas = pd.read_csv('data/df_unfallatlas.csv', delimiter=';', low_memory=False)
    df_unfallatlas['AGS'] = df_unfallatlas['AGS'].astype(str).str.zfill(8)
except:
    df_unfallatlas = get_data()
    df_unfallatlas = preprocessing(df_unfallatlas)
    df_unfallatlas = pred_IstGkfz(df_unfallatlas)
    df_unfallatlas.to_csv('data/df_unfallatlas.csv', sep=  ';', index=False)

#Einlesen und verarbeiten der Wetterdaten.
wheater_data_munich = get_wheater_data(wheater_data_munich)
#wheater_data_ger = get_wheater_data(wheater_data_ger)

#Hinzufügen der exogenen Daten.
#df_unfallatlas = add_exog_data(df_unfallatlas, wheater_data_ger)
print('Daten verarbeitet!')

#Auswahl der Vorhersage.
message = '\nWas wollen Sie tun?'
print(message, '\n')
tool = query_exception(dict = tools, message = message)

if tool == 0:
    print('\n#####################################')
    print('Vorhersage der schwere eines Unfalls.')
    print('#####################################\n')
    print('Erstellung des Modells ...\n')

    #Laden des Modells.
    try:
        model = joblib.load(f'models/{load_model_accident_severity}.sav')
    except:
        model = pred_accident_severity_nearest_neighbors(df_unfallatlas)

    prediction = query()
    accident_severity = model.predict(prediction)

    #Ausgabe der Unfallkategorie
    print('\nUnfallkategorie:\t', kategorien[accident_severity[0]])
    print('####################################################\n')

elif tool == 1:
    print('\n####################################################')
    print('Vorhersage der Anzahl der Unfälle für das Jahr 2021.')
    print('####################################################\n')

    #Erstellung des Modells mit vorheriger Grid Search zur Definition der besten Parameter für das Modell.
    df_number_of_accidents, wheater_data_munich_2021 = prepare_number_of_accidents(df_unfallatlas, ags, wheater_data_munich, visualization_mode)
    bestAIC, bestParam, bestSParam = grid_search(y = df_number_of_accidents['Count'], x = df_number_of_accidents[model_features])
    sarima = sarima(bestParam, bestSParam, visualization_mode, y = df_number_of_accidents['Count'], x = df_number_of_accidents[model_features])

    #Vorhersage der Anzahl der Unfälle für das Jahr 2021.
    pred_start, pred_end = str(np.min(df_number_of_accidents.index) + relativedelta(months = 48)), str(np.max(df_number_of_accidents.index) + relativedelta(months=12))
    prediction = sarima.get_prediction(pred_start, pred_end, exog=wheater_data_munich_2021[model_features])
    df_number_of_accidents = pd.concat([df_number_of_accidents['Count'], round(prediction.predicted_mean.last('12M'))])
    df_number_of_accidents = pd.DataFrame(df_number_of_accidents).rename(columns = {0: 'Count'})

    #Validierung des SARIMAX Modells anhand des MSE und MAE bezogen auf das Jahr 2020. Vergleich der Scores mit dem Modell ohne exogene Variablen.
    mse_value = round(mean_squared_error(df_number_of_accidents['Count'].loc[df_number_of_accidents.index.year == 2020], prediction.predicted_mean.head(12)), 2)
    mae_value = round(mean_absolute_error(df_number_of_accidents['Count'].loc[df_number_of_accidents.index.year == 2020], prediction.predicted_mean.head(12)), 2)

    print('\nVorhersage der Unfallzahlen auf Monatsbasis:')
    print(round(prediction.predicted_mean.last('12M')))

    visualization_ts(df_number_of_accidents, prediction)

    print('\nSie können sich nun eine Map anzeigen lassen, in welcher die geährlichsten Unfallstellen in München dargstellt sind.')

    message = 'Die gefärhrlichsten Unfallstellen können sie für einen bestimmten Monat betrachten oder für das ganze Jahr:'
    print(message, '\n')
    period_map = query_exception(dict = monate_map, message = message)

    # Vorbereitung des Datensatzes zur Darstellung der Unfallorte.
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