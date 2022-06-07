import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler, NearMiss


#Für Abfrage-Funktion.__________________________________________________________________________________________________

#Übersichtlichere Daten.
tools = {0: 'Vorhersage der Unfallkategorie.', 1: 'Vorhersage der Anzahl der Unfälle für das Jahr 2021.'}

monate = {
        1: 'Januar', 2: 'Februar', 3: 'März', 4: 'April', 5: 'Mai',
        6: 'Juni', 7: 'Juli', 8: 'August', 9: 'September',
        10: 'Oktober', 11: 'November', 12: 'Dezember'
        }

monate_map = {
        1: 'Januar', 2: 'Februar', 3: 'März', 4: 'April', 5: 'Mai',
        6: 'Juni', 7: 'Juli', 8: 'August', 9: 'September',
        10: 'Oktober', 11: 'November', 12: 'Dezember', 0: 'ganzjährig'
        }

wochentage = {1: 'Montag', 2: 'Dienstag', 3: 'Mittwoch', 4: 'Donnerstag', 5: 'Freitag', 6: 'Samstag', 7: 'Sonntag'}

arten = {
        1: 'Zusammenstoß mit anfahrendem/anhaltendem/ruhendem Fahrzeug',
        2: 'Zusammenstoß mit vorausfahrendem/wartendem Fahrzeug',
        3: 'Zusammenstoß mit seitlich in gleicher Richtung fahrendem Fahrzeug',
        4: 'Zusammenstoß mit entgegenkommendem Fahrzeug',
        5: 'Zusammenstoß mit einbiegendem/kreuzendem Fahrzeug',
        6: 'Zusammenstoß zwischen Fahrzeug und Fußgänger',
        7: 'Aufprall auf Fahrbahnhindernis',
        8: 'Abkommen von Fahrbahn nach rechts',
        9: 'Abkommen von Fahrbahn nach links',
        0: 'Unfall anderer Art',
        }

kategorien = {
        1: 'Unfall mit Getöteten       ',
        2: 'Unfall mit Schwerverletzten',
        3: 'Unfall mit Leichtverletzten',
        }

jaodernein = {
    0: 'Nein',
    1: 'Ja'
    }

#Funktion für die Abfrage mit Exception Handling.
def query_exception(dict, message):

    selection = None
    for key in dict.keys():
        print(f'{key} =', dict[key])

    while selection not in dict.keys():
        try:
            selection = int(input('\nEingabe: '))
            if selection not in dict.keys():
                print(message)
                print('Bitte wählen Sie eine der Optionen:\n')
                for key in dict.keys():
                    print(f'{key} =', dict[key])
        except ValueError:
            print(message)
            print('Bitte wählen Sie eine der Optionen:\n')
            for key in dict.keys():
                print(f'{key} =', dict[key])
    return selection

def query():

    #Gemeinde
    #ULAND = 9
    #UREGBEZ = 1
    #UKREIS = 62
    #UGEMEINDE = 0

    #Zeitpunkt
    now = datetime.datetime.now()
    UMONAT = now.month
    USTUNDE = now.hour
    UWOCHENTAG = now.isoweekday()

    #Was ist bei dem Unfall geschehen?
    message = '\nWas ist bei dem Unfall geschehen?'
    print(message, '\n')
    UART = query_exception(dict = arten, message = message)

    #Unfallteilnehmer
    message = '\nIst an dem Unfall ein Radfahrer beteiligt?'
    print(message, '\n')
    IstRad = query_exception(dict = jaodernein, message = message)

    message = '\nIst an dem Unfall ein PKW beteiligt?'
    print(message, '\n')
    IstPKW = query_exception(dict = jaodernein, message = message)

    message = '\nIst an dem Unfall ein Fußgänger beteiligt?'
    print(message, '\n')
    IstFuss = query_exception(dict = jaodernein, message = message)

    message = '\nIst an dem Unfall ein Kraftrad (Motorrad) beteiligt?'
    print(message, '\n')
    IstKrad = query_exception(dict = jaodernein, message = message)

    message = '\nIst an dem Unfall ein Güterkraftfahrzeug (LKW) beteiligt?'
    print(message, '\n')
    IstGkfz = query_exception(dict = jaodernein, message = message)

    message = '\nIst an dem Unfall ein Fahrzeug anderer Art beteiligt?'
    print(message, '\n')
    IstSonstige = query_exception(dict = jaodernein, message = message)

    print('\n###################################################')
    print('Zeitpunkt des Unfalls in München:')
    print('Unfallmonat:\t\t', monate[UMONAT])
    print('Unfallstunde:\t\t', USTUNDE, 'Uhr')
    print('Unfallwochentag:\t', wochentage[UWOCHENTAG])
    #print('\nUnfallart:\t', arten[UART])

    prediction = pd.DataFrame({#'ULAND': ULAND,
                               #'UREGBEZ': UREGBEZ,
                               #'UKREIS': UKREIS,
                               #'UGEMEINDE': UGEMEINDE,
                               'UMONAT': UMONAT,
                               'USTUNDE': USTUNDE,
                               'UWOCHENTAG': UWOCHENTAG,
                               'UART': UART,
                               'IstRad': IstRad,
                               'IstPKW': IstPKW,
                               'IstFuss': IstFuss,
                               'IstKrad': IstKrad,
                               'IstGkfz': IstGkfz,
                               'IstSonstige': IstSonstige,
                               }, index=[0])

    return prediction


#Für Vorhersage der schwere des Unfalls.________________________________________________________________________________

def visualization_ts(df_number_of_accidents, prediction):

    #Darstellung der TimeSeries.
    fig, ax = plt.subplots(figsize  = (15, 10))
    sns.lineplot(data = df_number_of_accidents['Number of Accidents'])
    sns.lineplot(data = prediction.predicted_mean)
    pred_ci = prediction.conf_int(0.1)
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=0.1)
    plt.title('Vorhersage der Anzahl der Unfälle', fontsize = 30, pad = 20)
    plt.ylabel('Anzahl der Unfälle', fontsize = 25)
    plt.legend(fontsize = 15, labels = ['Historische Werte', 'Vorhersage', 'Konfidenzband'], loc='upper left')
    ax.tick_params(axis = 'x', labelsize = 20)
    ax.tick_params(axis = 'y', labelsize = 20)
    plt.show()

    print('\n############################################################################################')
    print('In der obenstehenden Tabelle ist die prognostizierte Anzahl der Unfälle pro Monat angezeigt.')
    print('       In dem Plot ist die Prognose der Unfallzahlen für das Jahr 2021 dargestellt.')
    print('############################################################################################\n')


#Für Vorhersage der Unfallkategorie.____________________________________________________________________________________

def train_test_divid(df_unfallatlas, undersampling_mode):

    # Definition von X und y.
    X = df_unfallatlas.drop(['UKATEGORIE', 'lat', 'lon', 'UJAHR', 'UTYP1', 'AGS', 'ULICHTVERH', 'STRZUSTAND', 'ULAND', 'UREGBEZ', 'UGEMEINDE', 'UKREIS'], axis=1)
    y = df_unfallatlas['UKATEGORIE']

    # Skalierung der Attribute, für bessere Performance.
    scaler = StandardScaler().fit(X)
    X = pd.DataFrame(scaler.transform(X))

    # Splitten der Daten in Test- und Training-Set.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

    # Undersampling.
    X_train, y_train = undersampling(X_train, y_train, undersampling_mode)

    return X_train, X_test, y_train, y_test

def undersampling(X_train, y_train, undersampling_mode):

    # Undersampling.
    if undersampling_mode == 'random':
        rus = RandomUnderSampler(random_state=0)
        X_train, y_train = rus.fit_resample(X_train, y_train)
    elif undersampling_mode == 'nearmiss':
        nearmiss = NearMiss(version=3)
        X_train, y_train = nearmiss.fit_resample(X_train, y_train)

    return X_train, y_train