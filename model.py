import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import itertools
import statsmodels.api as sm
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV

warnings.simplefilter('ignore', ConvergenceWarning)

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

def visualization_ts(df_number_of_accidents, prediction):

    #Darstellung der TimeSeries.
    fig, ax = plt.subplots(figsize  = (15, 10))
    sns.lineplot(data = df_number_of_accidents['Count'])
    sns.lineplot(data = prediction.predicted_mean)
    pred_ci = prediction.conf_int(0.1)
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=0.1)
    plt.title('Vorhersage der Anzahl der Unfälle', fontsize = 30, pad = 20)
    plt.ylabel('Anzahl der Unfälle', fontsize = 25)
    plt.legend(fontsize = 15, labels = ['Historische Werte', 'Vorhersage', 'Konfidenzintervall'], loc='upper left')
    ax.tick_params(axis = 'x', labelsize = 20)
    ax.tick_params(axis = 'y', labelsize = 20)
    plt.show()

    print('\n############################################################################################')
    print('In der obenstehenden Tabelle ist die prognostizierte Anzahl der Unfälle pro Monat angezeigt.')
    print('       In dem Plot ist die Prognose der Unfallzahlen für das Jahr 2021 dargestellt.')
    print('############################################################################################\n')

#Vorhersage der schwere des Unfalls.____________________________________________________________________________________

def pred_accident_severity_decision_tree(df_unfallatlas):

    #Definition von X und y.
    X = df_unfallatlas.drop(['UKATEGORIE', 'lat', 'lon', 'UJAHR', 'UTYP1', 'AGS', 'ULICHTVERH', 'STRZUSTAND'], axis=1)
    y = df_unfallatlas['UKATEGORIE']

    #Splitten der Daten in Test- und Training-Set.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)

    #Training der Daten.
    decision_tree_clf = DecisionTreeClassifier(max_depth = 10, random_state=1).fit(X_train, y_train)

    #Validierung des Modells.
    results = pd.DataFrame(decision_tree_clf.predict(X_test), index = X_test.index)

    #Überprüfung der Genauigkeit des Modells.
    score = accuracy_score(y_test, results)
    #print('\nAccuracy-Score des Modells:', round(score, 2))

    return decision_tree_clf

def pred_accident_severity_nearest_neighbors(df_unfallatlas):

    #Definition von X und y.
    X = df_unfallatlas.drop(['UKATEGORIE', 'lat', 'lon', 'UJAHR', 'UTYP1', 'AGS', 'ULICHTVERH', 'STRZUSTAND'], axis=1)
    y = df_unfallatlas['UKATEGORIE']

    #Splitten der Daten in Test- und Training-Set.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.10)

    #Training der Daten.
    knn_clf = KNeighborsClassifier(n_neighbors = 1).fit(X_train, y_train)

    #Validierung des Modells.
    results = pd.DataFrame(knn_clf.predict(X_test), index = X_test.index)

    #Überprüfung der Genauigkeit des Modells.
    score = accuracy_score(y_test, results)
    #print('\nAccuracy-Score des Modells:', round(score, 2))

    return knn_clf

def grid_search_knn(df_unfallatlas):
    # Definition von X und y.
    X = df_unfallatlas.drop(['UKATEGORIE', 'lat', 'lon', 'UJAHR', 'UTYP1', 'AGS', 'ULICHTVERH', 'STRZUSTAND'], axis=1)
    y = df_unfallatlas['UKATEGORIE']

    #List Hyperparameters that we want to tune.
    leaf_size = list(range(1,10))
    n_neighbors = list(range(1,10))
    p=[1,2]

    #Convert to dictionary
    hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

    #Create new KNN object
    knn_2 = KNeighborsClassifier()

    #Use GridSearch
    clf = GridSearchCV(knn_2, hyperparameters, cv=10)

    #Fit the model
    best_model = clf.fit(X,y)

    #Print The value of best Hyperparameters
    print('Best leaf_size:', best_model.best_estimator_.get_params()['leaf_size'])
    print('Best p:', best_model.best_estimator_.get_params()['p'])
    print('Best n_neighbors:', best_model.best_estimator_.get_params()['n_neighbors'])