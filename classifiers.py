import pandas as pan
import matplotlib.pyplot as pyplt
import seaborn as sbrn
import numpy as np
import util as util
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import linear_model, neighbors, cluster
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from neupy import algorithms
from pyearth import Earth
import time
import warnings
warnings.filterwarnings("ignore")

def classifica(X_train,X_test,Y_train,Y_test):

    start = time.time()
    classifier = SVC()

    classifier.fit(X_train, Y_train)

    score = classifier.score(X_test, Y_test)

    print('SVM: '+str(score))

    classifier = linear_model.LinearRegression()
    classifier.fit(X_train, Y_train)
    y_pre = classifier.predict(X_test)
    ypre = np.around(y_pre)
    score = accuracy_score(Y_test, ypre)

    print('Regressao Linear: '+str(score))

    classifier = DecisionTreeClassifier( max_depth = 4 )

    classifier.fit( X_train, Y_train )
    score = classifier.score(X_test, Y_test)

    print('Arvore de Decisão: '+str(score))

    classifier = GaussianNB()

    classifier.fit( X_train, Y_train )
    score = classifier.score(X_test, Y_test)

    print('Bayes: '+str(score))

    classifier = MLPClassifier(hidden_layer_sizes=(25,25,25))

    classifier.fit( X_train, Y_train )
    score = classifier.score(X_test, Y_test)

    print('Rede Neural: '+str(score))

    classifier = neighbors.KNeighborsClassifier()

    classifier.fit( X_train, Y_train )
    score = classifier.score(X_test, Y_test)

    print('KNN: '+str(score))

    classifier = cluster.KMeans(n_clusters=2)
    classifier.fit_transform(X_train)
    labelspre = classifier.predict(X_test)
    score = accuracy_score(Y_test, labelspre)

    print('Cluster: '+str(score))

    classifier = linear_model.LogisticRegression()
    classifier.fit(X_train, Y_train)
    y_pre = classifier.predict(X_test)
    score = accuracy_score(Y_test, y_pre)

    print('Regressão Logística: '+str(score))

    classifier = LinearDiscriminantAnalysis()
    classifier.fit(X_train, Y_train)
    score = classifier.score(X_test, Y_test)

    print('LDA: '+str(score))
    classifier = algorithms.GRNN(std=1, verbose=False)
    classifier.fit(X_train, Y_train)
    y_pre = classifier.predict(X_test)
    ypre = np.around(y_pre)
    score = accuracy_score(Y_test, ypre)

    print('GRNN: '+str(score))

    classifier = Earth()
    classifier.fit(X_train, Y_train)
    y_pre = classifier.predict(X_test)
    ypre = np.around(y_pre)
    score = accuracy_score(Y_test, ypre)

    print('MARS: '+str(score))
    end = time.time()
    print('Tempo de execução:' + str(end - start))

def classificaPCA(X_train,X_test,Y_train,Y_test,components):

    pca = PCA(n_components=components)
    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)
    start = time.time()
    classifier = SVC()

    classifier.fit(X_train, Y_train)

    score = classifier.score(X_test, Y_test)

    print('SVM+PCA: '+str(score))

    classifier = linear_model.LinearRegression()
    classifier.fit(X_train, Y_train)
    y_pre = classifier.predict(X_test)
    ypre = np.around(y_pre)
    score = accuracy_score(Y_test, ypre)

    print('Regressao Linear+PCA: '+str(score))

    classifier = DecisionTreeClassifier( max_depth = 4 )

    classifier.fit( X_train, Y_train )
    score = classifier.score(X_test, Y_test)

    print('Arvore de Decisão+PCA: '+str(score))

    classifier = GaussianNB()

    classifier.fit( X_train, Y_train )
    score = classifier.score(X_test, Y_test)

    print('Bayes+PCA: '+str(score))

    classifier = MLPClassifier(hidden_layer_sizes=(25,25,25))

    classifier.fit( X_train, Y_train )
    score = classifier.score(X_test, Y_test)

    print('Rede Neural+PCA: '+str(score))

    classifier = neighbors.KNeighborsClassifier()

    classifier.fit( X_train, Y_train )
    score = classifier.score(X_test, Y_test)

    print('KNN+PCA: '+str(score))

    classifier = cluster.KMeans(n_clusters=2)
    classifier.fit_transform(X_train)
    labelspre = classifier.predict(X_test)
    score = accuracy_score(Y_test, labelspre)

    print('Cluster+PCA: '+str(score))

    classifier = linear_model.LogisticRegression()
    classifier.fit(X_train, Y_train)
    y_pre = classifier.predict(X_test)
    score = accuracy_score(Y_test, y_pre)

    print('Regressão Logística+PCA: '+str(score))

    classifier = LinearDiscriminantAnalysis()
    classifier.fit(X_train, Y_train)
    score = classifier.score(X_test, Y_test)

    print('LDA+PCA: '+str(score))
    classifier = algorithms.GRNN(std=1, verbose=False)
    classifier.fit(X_train, Y_train)
    y_pre = classifier.predict(X_test)
    ypre = np.around(y_pre)
    score = accuracy_score(Y_test, ypre)

    print('GRNN+PCA: '+str(score))

    classifier = Earth()
    classifier.fit(X_train, Y_train)
    y_pre = classifier.predict(X_test)
    ypre = np.around(y_pre)
    score = accuracy_score(Y_test, ypre)

    print('MARS+PCA: '+str(score))
    end = time.time()
    print('Tempo de execução:' + str(end - start))