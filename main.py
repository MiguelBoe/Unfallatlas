import pandas as pd
import folium
from folium import plugins
import webbrowser
from data_preprocessing import get_data, preprocessing, pred_IstGkfz
from utils import query_exception, query, kategorien, tools
from model import sarima, pred_accident_severity_decision_tree, pred_accident_severity_nearest_neighbors, \
                  pred_number_of_accidents, grid_search, visualization_ts, grid_search_knn

#Parameter
selection = None
ags = '09162000'

#Begrüßung.
print('\n##################################################')
print('Willkommen beim Unfallvorhersage-Tool für München!')
print('##################################################\n')
print('Einlesen und Verarbeiten der Daten ...')

#Einlesen und Verarbeiten der Daten.
df_unfallatlas = get_data()
df_unfallatlas = preprocessing(df_unfallatlas)
df_unfallatlas = pred_IstGkfz(df_unfallatlas)
print('Daten verarbeitet!')

#Auswahl der Vorhersage.
message = '\nWas wollen Sie tun?'
print(message, '\n')
tool = query_exception(dict = tools, message= message)

if tool == 0:
    print('\n#####################################')
    print('Vorhersage der schwere eines Unfalls.')
    print('#####################################\n')
    print('Erstellung des Modells ...\n')

    #grid_search_knn(df_unfallatlas)
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
    df_number_of_accidents = pred_number_of_accidents(df_unfallatlas, ags)
    bestAIC, bestParam, bestSParam = grid_search(y = df_number_of_accidents)
    sarima = sarima(bestParam, bestSParam, y = df_number_of_accidents)

    #Vorhersage der Anzahl der Unfälle für das Jahr 2021.
    prediction = sarima.get_forecast(steps = 12)
    df_number_of_accidents = pd.concat([pd.Series(df_number_of_accidents), pd.Series(round(prediction.predicted_mean))])
    df_number_of_accidents = pd.DataFrame(df_number_of_accidents).rename(columns = {0: 'Count'})
    visualization_ts(df_number_of_accidents, prediction)

    print('\n############################################################################')
    print('In dem Plot ist die Prognose der Unfallzahlen für das Jahr 2021 dargestellt.')
    print('In der obenstehenden Tabelle sind die prognostizierten Unfallzahlen angezeigt.')
    print('   In der Map sind die gefährlichsten Unfallorte in München dargestellt.')
    print('############################################################################\n')

    #Vorbereitung des Datensatzes zur Darstellung der Unfallorte.
    df_unfallatlas_visualization = df_unfallatlas[(df_unfallatlas['AGS'] == ags) & (df_unfallatlas['UKATEGORIE'] != 3)].reset_index(drop = True)

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
            #for point in range(0, len(locationlist)):
            #    folium.CircleMarker(locationlist[point], popup = df_unfallatlas_visualization['UART'][point], radius = 3).add_to(accident_map)
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