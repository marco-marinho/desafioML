import pandas as pan
import matplotlib.pyplot as pyplt
import seaborn as sbrn
import numpy as np
import util as util
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import linear_model, neighbors, cluster
from sklearn.metrics import accuracy_score, log_loss
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
    Y_Hat = classifier.predict(X_test)

    score = log_loss(Y_test, Y_Hat)

    print('SVM: '+str(score))

    classifier = linear_model.LinearRegression()
    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    Y_Hat = np.around(Y_Hat)
    score = log_loss(Y_test, Y_Hat)

    print('Regressao Linear: '+str(score))

    classifier = DecisionTreeClassifier( max_depth = 4 )

    classifier.fit( X_train, Y_train )
    Y_Hat = classifier.predict(X_test)

    score = log_loss(Y_test, Y_Hat)

    print('Arvore de Decisão: '+str(score))

    classifier = GaussianNB()

    classifier.fit( X_train, Y_train )
    Y_Hat = classifier.predict(X_test)

    score = log_loss(Y_test, Y_Hat)

    print('Bayes: '+str(score))

    classifier = MLPClassifier(hidden_layer_sizes=(25,25,25))

    classifier.fit( X_train, Y_train )
    Y_Hat = classifier.predict(X_test)

    score = log_loss(Y_test, Y_Hat)

    print('Rede Neural: '+str(score))

    classifier = neighbors.KNeighborsClassifier()

    classifier.fit( X_train, Y_train )
    Y_Hat = classifier.predict(X_test)
    score = log_loss(Y_test, Y_Hat)

    print('KNN: '+str(score))

    classifier = cluster.KMeans(n_clusters=2)
    classifier.fit_transform(X_train)
    Y_Hat = classifier.predict(X_test)
    score = log_loss(Y_test, Y_Hat)

    print('Cluster: '+str(score))

    classifier = linear_model.LogisticRegression()
    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    score = log_loss(Y_test, Y_Hat)

    print('Regressão Logística: '+str(score))

    classifier = LinearDiscriminantAnalysis()
    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    score = log_loss(Y_test, Y_Hat)

    print('LDA: '+str(score))
    classifier = algorithms.GRNN(std=1, verbose=False)
    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    Y_Hat = np.around(Y_Hat)
    score = log_loss(Y_test, Y_Hat)

    print('GRNN: '+str(score))

    classifier = Earth()
    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    Y_Hat = np.around(Y_Hat)
    score = log_loss(Y_test, Y_Hat)

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
    Y_Hat = classifier.predict(X_test)

    score = log_loss(Y_test, Y_Hat)

    print('SVM: '+str(score))

    classifier = linear_model.LinearRegression()
    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    Y_Hat = np.around(Y_Hat)
    score = log_loss(Y_test, Y_Hat)

    print('Regressao Linear: '+str(score))

    classifier = DecisionTreeClassifier( max_depth = 4 )

    classifier.fit( X_train, Y_train )
    Y_Hat = classifier.predict(X_test)

    score = log_loss(Y_test, Y_Hat)

    print('Arvore de Decisão: '+str(score))

    classifier = GaussianNB()

    classifier.fit( X_train, Y_train )
    Y_Hat = classifier.predict(X_test)

    score = log_loss(Y_test, Y_Hat)

    print('Bayes: '+str(score))

    classifier = MLPClassifier(hidden_layer_sizes=(25,25,25))

    classifier.fit( X_train, Y_train )
    Y_Hat = classifier.predict(X_test)

    score = log_loss(Y_test, Y_Hat)

    print('Rede Neural: '+str(score))

    classifier = neighbors.KNeighborsClassifier()

    classifier.fit( X_train, Y_train )
    Y_Hat = classifier.predict(X_test)
    score = log_loss(Y_test, Y_Hat)

    print('KNN: '+str(score))

    classifier = cluster.KMeans(n_clusters=2)
    classifier.fit_transform(X_train)
    Y_Hat = classifier.predict(X_test)
    score = log_loss(Y_test, Y_Hat)

    print('Cluster: '+str(score))

    classifier = linear_model.LogisticRegression()
    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    score = log_loss(Y_test, Y_Hat)

    print('Regressão Logística: '+str(score))

    classifier = LinearDiscriminantAnalysis()
    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    score = log_loss(Y_test, Y_Hat)

    print('LDA: '+str(score))
    classifier = algorithms.GRNN(std=1, verbose=False)
    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    Y_Hat = np.around(Y_Hat)
    score = log_loss(Y_test, Y_Hat)

    print('GRNN: '+str(score))

    classifier = Earth()
    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    Y_Hat = np.around(Y_Hat)
    score = log_loss(Y_test, Y_Hat)

    print('MARS: '+str(score))
    end = time.time()
    print('Tempo de execução:' + str(end - start))