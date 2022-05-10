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
from pylab import rcParams
import matplotlib
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
import statsmodels
from sklearn.model_selection import GridSearchCV

warnings.simplefilter('ignore', ConvergenceWarning)

#SARIMA.________________________________________________________________________________________________________________

#Vorbereitung der Zeireihe und kurze Analyse.
def pred_number_of_accidents(df_unfallatlas, ags):

    #Index to datetime. Allerdings Problem wegen des Tages. Dieser ist ja nicht angegeben. Habe für Testzwecke mal den Wochentag genommen.
    df_number_of_accidents = df_unfallatlas[(df_unfallatlas['AGS'] == ags)].reset_index(drop = True)
    df_number_of_accidents.rename(columns={'UJAHR': 'year', 'UMONAT': 'month', 'UWOCHENTAG': 'day', 'UKATEGORIE': 'Count'},inplace=True)
    df_number_of_accidents = df_number_of_accidents.set_index(pd.to_datetime(df_number_of_accidents[['year', 'month', 'day']])).resample('M')['Count'].count()
    df_number_of_accidents.index = df_number_of_accidents.index.map(lambda t: t.replace(day = 1))
    df_number_of_accidents.index.freq = 'MS'

    #Plot Dekomposition.
    rcParams['figure.figsize'] = 15, 10
    decomposition = sm.tsa.seasonal_decompose(df_number_of_accidents, model = 'additive')
    fig = decomposition.plot()
    plt.show()

    #Plot ACF und PACF.
    plot_acf(df_number_of_accidents)
    matplotlib.pyplot.show()
    plot_pacf(df_number_of_accidents, method='ywm')
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

    return df_number_of_accidents

#Algorithmus zur Bestimmung der optimalen Parameterkombination.
def grid_search(y):

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
                                                enforce_invertibility = False)
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
def sarima(bestParam, bestSParam, y):

    sarima = sm.tsa.statespace.SARIMAX(y,
                                       order = bestParam,
                                       seasonal_order = bestSParam,
                                       enforce_stationarity = False,
                                       enforce_invertibility = False).fit(disp = 0)

    #Darstellung der Diagnosedaten des Modells.
    sarima.plot_diagnostics(figsize = (18, 8))
    plt.show()

    return sarima

def visualization_ts(df_number_of_accidents, prediction):

    #Darstellung der TimeSeries.
    fig, ax = plt.subplots(figsize  = (15, 10))
    sns.lineplot(data = df_number_of_accidents['Count'])
    sns.lineplot(data = prediction.predicted_mean)
    plt.title('Vorhersage der Anzahl der Unfälle', fontsize = 30, pad = 20)
    plt.ylabel('Anzahl der Unfälle', fontsize = 25)
    plt.legend(fontsize = 15, title_fontsize = 15, labels = ['Historische Werte', 'Vorhersage'], bbox_to_anchor = (1.02, 1), loc = 2, borderaxespad = 0.)
    ax.tick_params(axis = 'x', labelsize = 20)
    ax.tick_params(axis = 'y', labelsize = 20)
    plt.show()

    print('Vorhersage der Unfallzahlen auf Monatsbasis:')
    print(round(prediction.predicted_mean))

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