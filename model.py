import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import itertools
import statsmodels.api as sm
from pylab import rcParams
import matplotlib
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from sklearn import svm
import joblib

warnings.simplefilter('ignore', ConvergenceWarning)


#Vorhersage der schwere des Unfalls.____________________________________________________________________________________

def pred_accident_severity(df_unfallatlas):

    #Definition von X und y.
    X = df_unfallatlas.drop(['UKATEGORIE', 'lat', 'lon'], axis=1)
    y = df_unfallatlas['UKATEGORIE']

    #Splitten der Daten in Test- und Training-Set.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    #Training der Daten.
    decision_tree_classification = DecisionTreeClassifier(max_depth=10, random_state=1).fit(X_train, y_train)
    joblib.dump(decision_tree_classification, '/test/')

    #Validierung des Modells.
    results = pd.DataFrame(decision_tree_classification.predict(X_test), index=X_test.index)

    #Überprüfung der Genauigkeit des Modells.
    score = accuracy_score(y_test, results)
    print('\nAccuracy-Score des Modells:', round(score, 2))

    return decision_tree_classification

def pred_accident_severity_svm(df_unfallatlas):

    #Vorhersage der Unfallkategorie mit SVM
    X_SVM = df_unfallatlas.drop(['UKATEGORIE', 'lat', 'lon'], axis=1)
    y_SVM = df_unfallatlas['UKATEGORIE']

    #Splitten der Daten in Test- und Training-Set.
    X_SVM_train, X_SVM_test, y_SVM_train, y_SVM_test = train_test_split(X_SVM, y_SVM, test_size=0.20)

    #Generierung svm-Classifier
    clf = svm.SVC(kernel = 'rbf')

    #Training der Daten.
    SVM = clf.fit(X_SVM_train, y_SVM_train)

    #Validierung des Modells.
    results_SVM = pd.DataFrame(clf.predict(X_SVM_test), index=X_SVM_test.index)

    #Überprüfung der Genauigkeit des Modells.
    score_SVM = accuracy_score(y_SVM_test, results_SVM)
    print('\nAccuracy-Score des Modells SVM:', round(score, 2))

    return results_SVM


#SARIMA.________________________________________________________________________________________________________________

#Vorbereitung der Zeireihe und kurze Analyse.
def pred_number_of_accidents(df_unfallatlas, ags):

    #Index to datetime. Allerdings Problem wegen des Tages. Dieser ist ja nicht angegeben. Habe für Testzwecke mal den Wochentag genommen.
    df_number_of_accidents = df_unfallatlas[(df_unfallatlas['AGS'] == ags)].reset_index(drop = True)
    df_number_of_accidents.rename(columns={'UJAHR': 'year', 'UMONAT': 'month', 'UWOCHENTAG': 'day', 'UKATEGORIE': 'Count'},inplace=True)
    df_number_of_accidents = df_number_of_accidents.set_index(pd.to_datetime(df_number_of_accidents[['year', 'month', 'day']])).resample('M')['Count'].count()

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
                                                order=param,
                                                seasonal_order=param_seasonal,
                                                enforce_stationarity=False,
                                                enforce_invertibility=False)
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
    fig, ax = plt.subplots(figsize  =(15, 10))
    sns.lineplot(data = df_number_of_accidents)
    sns.lineplot(data = prediction.predicted_mean)
    plt.title('Vorhersage der Anzahl der Unfälle', fontsize = 30, pad = 20)
    plt.ylabel('Anzahl der Unfälle', fontsize = 25)
    plt.legend(fontsize = 15, title_fontsize = 15, labels = ['Historische Werte', 'Vorhersage'], bbox_to_anchor = (1.02, 1), loc = 2, borderaxespad = 0.)
    ax.tick_params(axis = 'x', labelsize = 20)
    ax.tick_params(axis = 'y', labelsize = 20)
    plt.show()











