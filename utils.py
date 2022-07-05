import pandas as pd
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler, NearMiss


#Für Abfrage-Funktion.__________________________________________________________________________________________________

'''
Erstellung verschiedener Dictionaries für ein besseres Verständnis über die Daten. Mit diesen Dictionaries und den nach-
folgenden Funktionen wurde die interaktive Anwendung erstellt.
'''
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

'''
Die Funktion query_exception() ist für den Ablauf der interaktiven Anwendung besonders wichtig. Diese Funktion wird in dem 
gesamten Abfrage-Prozess für die Vorhersage der Unfallkategorie verwendet. Der Funktion wird immer eine "message" übergeben, 
also eine Abfrage und ein entsprechendes Dictionary. Die Funktion stellt dann sicher, dass der Nutzer auch eine zulässige 
Eingabe tätigt. Ist dies nicht der Fall, wird die Abfrage wiederholt, mit dem Hinweis, dass der Nutzer nur eine der angegebenen 
Optionen auswählen kann. Zudem werden die genannten Auswahloptionen erneut präsentiert.
'''
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

'''
Die Funktion query() koordiniert die Abfrage, wobei die Informationen (Attribute) für die Prognose der Unfallkategorie vom
Nutzer ermittelt werden. Die Informationen über den Zeitpunkt des Unfalls werden automatisch bei der Nutzung der Funktion
ermittelt. Diese beziehen sich auf den aktuellen Zeitpunkt. Als nächstes wird abgefragt, was bei dem Unfall geschehen ist.
Dafür wird die Funktion query_exception() aufgerufen und die "message" sowie das entsprechende Dictionary dazu übergeben.
Anschließend wird nach dem selben Prinzip abgefragt, wer an dem Unfall beteiligt ist. Nach der Abfrage der Informationen
werden diese in einem DataFrame abgespeichert. Mit diesem wird in einem nächsten Schritt die Unfallkategorie prognostiziert.
Davor wird jedoch zunächst eine kurze Zusammenfassung der eingegebene Informationen ausgegeben, wonach dann die Unfallkategorie,
basierend auf der Vorhersage des Modells, angegeben wird.
'''
def query():

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

    prediction = pd.DataFrame({'UMONAT': UMONAT,
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

'''
Die Funktion visualization_ts() ist für die Visualisierung einer Zeitreihe (englisch: time series (ts)) von Nutzen. Diese
Zeitreihe wird auf Basis des DataFrames df_number_of_accidents dargestellt, welcher für die Prognose der Unfallzahlen er-
stellt wurde. Insgesamt wird in der Funktion einmal die Zeitreihe mit den vorhandenen Ist-Werten dargestellt (ab 2016) und
die prognostizierte Zeitreihe von 2020 bis 2022 (Test-Set von 2020 bis 2021) dargestellt. Zudem wurde für die prognostizierte 
Zeitreihe das Konfidenzband mit dargestellt.
'''
def visualization_ts(df_number_of_accidents, prediction):

    #Darstellung der TimeSeries.
    fig, ax = plt.subplots(figsize  = (15, 10))
    sns.lineplot(data = df_number_of_accidents['Number of Accidents'])
    sns.lineplot(data = prediction.predicted_mean)
    pred_ci = prediction.conf_int(alpha=0.1)
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

'''
Die Funktion train_test_divid() führt den Split des Datensatzes bei der Entwicklung der Modelle für die Vorhersage der Unfall-
kategorie aus. Da mehrere Modelle trainiert wurden und dieser Split bei jedem Modell identisch ist, wurde die Funktion ausgelagert,
damit diese immer wieder verwendet werden kann. Somit wird auch der Code etwas kürzer. Ein wichtiger Punkt in dieser Funktion ist
der StandardScaler, welcher benutzt wird, um die Daten zu normalisieren. Dies dient in erster Linie für das Training der SVM.
Allerdings schadet die Normalisierung nicht bei dem Training der anderen Modelle. Weshalb diese Normalisierung bei dem Training
jedes Modells angwendet wird. Zudem wird in der train_test_divid()-Funktion noch die undersampling()-Funktion aufgerufen, 
welche nachfolgend beschrieben wird.
'''
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

'''
Da die Datenpunkte für die Vorhersage der Unfallkategorie unausgeglichen sind, wurde diese undersampling()-Funktion implementiert,
um eine Gleichverteilung bezüglich der Unfallkategorie zu erzeugen. Der Funktion wird neben den Daten der undersampling_mode-
Parameter übergeben. Dieser wird im Konfigurationsbereich definiert und es kann damit gesteuert werden, wie das Undersampling
erfolgen soll. Das Undersampling kann jedoch auch komplett ausgeschaltet werden, indem der Parameter auf False gesetzt wird.
'''
def undersampling(X_train, y_train, undersampling_mode):

    # Undersampling.
    if undersampling_mode == 'random':
        rus = RandomUnderSampler(random_state=0)
        X_train, y_train = rus.fit_resample(X_train, y_train)
    elif undersampling_mode == 'nearmiss':
        nearmiss = NearMiss(version=3)
        X_train, y_train = nearmiss.fit_resample(X_train, y_train)

    return X_train, y_train