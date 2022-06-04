import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import itertools
import statsmodels.api as sm
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from utils import undersampling, train_test_divid
import joblib

warnings.simplefilter('ignore', ConvergenceWarning)
warnings.simplefilter('ignore', FutureWarning)


#SARIMA.________________________________________________________________________________________________________________

#Algorithmus zur Bestimmung der optimalen Parameterkombination.
def grid_search(y, x):

    p = d = q = range(0, 2)

    pdq = list(itertools.product(p, d, q))
    seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]

    bestAIC = np.inf
    bestParam = None
    bestSParam = None

    print('\nAusführung GridSearch ...\n')

    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mod = sm.tsa.statespace.SARIMAX(y,
                                                order = param,
                                                seasonal_order = param_seasonal,
                                                enforce_stationarity = False,
                                                enforce_invertibility = False, exog=x)
                results = mod.fit(maxiter=200, disp=0)
                print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
                if results.aic < bestAIC:
                    bestAIC = results.aic
                    bestParam = param
                    bestSParam = param_seasonal
            except:
                continue

    print('\nDie besten Parameter sind:\n')
    print('ARIMA{}x{}12 - AIC:{}'.format(bestParam, bestSParam, bestAIC))
    print('\nErstellung eines Modells mit den besten Parametern ...\n')

    return bestAIC, bestParam, bestSParam

#Erstellung des Modells.
def sarima(bestParam, bestSParam, visualization_mode, y, x):

    sarima = sm.tsa.statespace.SARIMAX(y,
                                       order = bestParam,
                                       seasonal_order = bestSParam,
                                       enforce_stationarity = False,
                                       enforce_invertibility = False, exog=x).fit(disp = 0)

    #Darstellung der Diagnosedaten des Modells.
    if visualization_mode:
        sarima.plot_diagnostics(figsize = (18, 8))
        plt.show()

    return sarima


#Vorhersage der Unfallkategorie.________________________________________________________________________________________

#Naives statistisches Mehrheitsverfahren ohne Undersampling als Baseline Modell.
def baseline_model(df_unfallatlas):

    # Definition von X und y.
    X = df_unfallatlas.drop(['UKATEGORIE', 'lat', 'lon', 'UJAHR', 'UTYP1', 'AGS', 'ULICHTVERH', 'STRZUSTAND', 'ULAND', 'UREGBEZ', 'UGEMEINDE', 'UKREIS'], axis=1)
    y = df_unfallatlas['UKATEGORIE']

    # Splitten der Daten in Test- und Training-Set.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

    # Vorbereitung der Daten.
    X_test = X_test.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    prediction = pd.Series()

    for row in range(0,len(X_test)):

        accident_severity = df_unfallatlas[(df_unfallatlas['UMONAT'] == X_test['UMONAT'].iloc[row]) &
              (df_unfallatlas['USTUNDE'] == X_test['USTUNDE'].iloc[row]) &
              (df_unfallatlas['UWOCHENTAG'] == X_test['UWOCHENTAG'].iloc[row]) &
              (df_unfallatlas['UART'] == X_test['UART'].iloc[row]) &
              (df_unfallatlas['IstRad'] == X_test['IstRad'].iloc[row]) &
              (df_unfallatlas['IstPKW'] == X_test['IstPKW'].iloc[row]) &
              (df_unfallatlas['IstFuss'] == X_test['IstFuss'].iloc[row]) &
              (df_unfallatlas['IstKrad'] == X_test['IstKrad'].iloc[row]) &
              (df_unfallatlas['IstGkfz'] == X_test['IstGkfz'].iloc[row]) &
              (df_unfallatlas['IstSonstige'] == X_test['IstSonstige'].iloc[row])]

        # Bestimmung der Unfallkategorie mit der höchsten Wahrscheinlichkeit.
        print(f'Prediction {row}' + f'/{len(X_test)-1}:', accident_severity['UKATEGORIE'].mode()[0])
        prediction = prediction.append(pd.Series(accident_severity['UKATEGORIE'].mode()[0]))

    # Erstellung des DataFrames mit den Ergebnissen der Prognose im Vergleich zu y_test.
    prediction = prediction.reset_index(drop=True)
    results = pd.DataFrame({'prediction':prediction, 'y_test': y_test})

    # Überprüfung der Genauigkeit des Modells.
    score = accuracy_score(results['y_test'], results['prediction'])
    baseline_clf_report = classification_report(results['y_test'], results['prediction'])

    return baseline_clf_report

def statistical_determination_accident_severity(df_unfallatlas, prediction):

    accident_severity = df_unfallatlas[(df_unfallatlas['UMONAT'] == prediction['UMONAT'][0]) &
          (df_unfallatlas['USTUNDE'] == prediction['USTUNDE'][0]) &
          (df_unfallatlas['UWOCHENTAG'] == prediction['UWOCHENTAG'][0]) &
          (df_unfallatlas['UART'] == prediction['UART'][0]) &
          (df_unfallatlas['IstRad'] == prediction['IstRad'][0]) &
          (df_unfallatlas['IstPKW'] == prediction['IstPKW'][0]) &
          (df_unfallatlas['IstFuss'] == prediction['IstFuss'][0]) &
          (df_unfallatlas['IstKrad'] == prediction['IstKrad'][0]) &
          (df_unfallatlas['IstGkfz'] == prediction['IstGkfz'][0]) &
          (df_unfallatlas['IstSonstige'] == prediction['IstSonstige'][0])]

    return accident_severity

def pred_accident_severity_decision_tree(df_unfallatlas, undersampling_mode):

    #Splitten der Daten in Test- und Training-Set.
    X_train, X_test, y_train, y_test = train_test_divid(df_unfallatlas, undersampling_mode)

    #Training der Daten.
    decision_tree_model = DecisionTreeClassifier(max_depth = 10, class_weight={1:0.40, 2:0.40, 3:0.20}).fit(X_train, y_train)
    #joblib.dump(decision_tree_model, 'models/decision_tree_model.sav')

    #Validierung des Modells.
    results = pd.DataFrame(decision_tree_model.predict(X_test), index = X_test.index)
    results['y_test'] = y_test

    #Überprüfung der Genauigkeit des Modells.
    score = accuracy_score(results['y_test'], results[0])
    clf_report = classification_report(results['y_test'], results[0])

    return decision_tree_model

def pred_accident_severity_random_forest(df_unfallatlas, undersampling_mode):

    #Splitten der Daten in Test- und Training-Set.
    X_train, X_test, y_train, y_test = train_test_divid(df_unfallatlas, undersampling_mode)

    #Training der Daten.
    random_forest_model = RandomForestClassifier(max_depth = 10, class_weight={1:0.40, 2:0.40, 3:0.20}).fit(X_train, y_train)
    #joblib.dump(random_forest_model, 'models/random_forest_model.sav')

    #Validierung des Modells.
    results = pd.DataFrame(random_forest_model.predict(X_test), index = X_test.index)
    results['y_test'] = y_test

    #Überprüfung der Genauigkeit des Modells.
    score = accuracy_score(results['y_test'], results[0])
    clf_report = classification_report(results['y_test'], results[0])

    return random_forest_model

def pred_accident_severity_gaussian_nb(df_unfallatlas, undersampling_mode):

    #Splitten der Daten in Test- und Training-Set.
    X_train, X_test, y_train, y_test = train_test_divid(df_unfallatlas, undersampling_mode)

    #Training der Daten.
    gaussian_nb_model = GaussianNB().fit(X_train, y_train)
    joblib.dump(gaussian_nb_model, 'models/gaussian_nb_model.sav')

    #Validierung des Modells.
    results = pd.DataFrame(gaussian_nb_model.predict(X_test), index = X_test.index)
    results['y_test'] = y_test

    #Überprüfung der Genauigkeit des Modells.
    score = accuracy_score(results['y_test'], results[0])
    clf_report = classification_report(results['y_test'], results[0])

    return gaussian_nb_model

def pred_accident_severity_svm(df_unfallatlas, undersampling_mode):

    # Splitten der Daten in Test- und Training-Set.
    X_train, X_test, y_train, y_test = train_test_divid(df_unfallatlas, undersampling_mode)

    # Skalierung der Attribute, für bessere Performance.
    scaler = StandardScaler().fit(X)
    X = pd.DataFrame(scaler.transform(X))

    # Splitten der Daten in Test- und Training-Set.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

    # Undersampling.
    X_train, y_train = undersampling(X_train, y_train, undersampling_mode)

    # Training der Daten.
    svm_model = svm.SVC(kernel='poly', class_weight={1:0.40, 2:0.40, 3:0.20}).fit(X_train, y_train)
    joblib.dump(svm_model, 'models/svm_model.sav')

    # Validierung des Modells.
    results = pd.DataFrame(svm_model.predict(X_test), index=X_test.index)
    results['y_test'] = y_test

    # Überprüfung der Genauigkeit des Modells.
    score = accuracy_score(results['y_test'], results[0])
    clf_report = classification_report(results['y_test'], results[0])

    return svm_model

def pred_accident_severity_nearest_neighbors(df_unfallatlas, undersampling_mode):

    # Splitten der Daten in Test- und Training-Set.
    X_train, X_test, y_train, y_test = train_test_divid(df_unfallatlas, undersampling_mode)

    #Training der Daten.
    knn_model = KNeighborsClassifier(n_neighbors = 3).fit(X_train, y_train)
    joblib.dump(knn_model, 'models/knn_model.sav')

    #Validierung des Modells.
    results = pd.DataFrame(knn_model.predict(X_test), index = X_test.index)
    results['y_test'] = y_test

    #Überprüfung der Genauigkeit des Modells.
    score = accuracy_score(results['y_test'], results[0])
    clf_report = classification_report(results['y_test'], results[0])

    return knn_model