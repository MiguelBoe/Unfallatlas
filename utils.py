import pandas as pd
import datetime

#Übersichtlichere Daten.
tools = {0: 'Vorhersage der Schwere eines Unfalls.', 1: 'Vorhersage der Anzahl der Unfälle für das Jahr 2021.'}

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
        1: 'Unfall mit Getöteten',
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
    ULAND = 9
    UREGBEZ = 1
    UKREIS = 62
    UGEMEINDE = 0

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

    print('\n####################################################')
    print('Zeitpunkt des Unfalls in München:')
    print('Unfallmonat:\t\t', monate[UMONAT])
    print('Unfallstunde:\t\t', USTUNDE, 'Uhr')
    print('Unfallwochentag:\t', wochentage[UWOCHENTAG])

    prediction = pd.DataFrame({'ULAND': ULAND,
                               'UREGBEZ': UREGBEZ,
                               'UKREIS': UKREIS,
                               'UGEMEINDE': UGEMEINDE,
                               'UMONAT': UMONAT,
                               'USTUNDE': USTUNDE,
                               'UWOCHENTAG': UWOCHENTAG,
                               'UART': UART,
                               'IstRad': IstRad,
                               'IstPKW': IstPKW,
                               'IstFuss': IstFuss,
                               'IstKrad': IstKrad,
                               'IstGkfz': IstGkfz,
                               'IstSonstige': IstSonstige
                               }, index=[0])

    return prediction
