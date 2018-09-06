import numpy as np
from sklearn.svm import SVC
from sklearn import linear_model, neighbors, cluster
from sklearn.metrics import accuracy_score, log_loss
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from neupy import algorithms
from pyearth import Earth
import time
import warnings
import util as utl

warnings.filterwarnings("ignore")


# Treina os métodos de classificação e mede o desempenho dos mesmos sem utilizar o PCA
def classifica(X_train, X_test, Y_train, Y_test):
    start = time.time()
    classifier = SVC(probability=True)

    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    end = time.time()
    Y_Hat_prob = classifier.predict_proba(X_test)

    score = accuracy_score(Y_test, Y_Hat)
    loss = log_loss(Y_test, Y_Hat_prob)
    print('SVM: ')
    print('Precisão: ' + str(round(score, 4)) + ' Logloss: ' + str(round(loss, 4)))
    print('Tempo de execução:' + str(end - start))

    start = time.time()

    classifier = linear_model.LinearRegression()
    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    end = time.time()
    Y_Hat_prob = utl.get_probability_round_aproximation(Y_Hat)
    Y_Hat = np.around(Y_Hat)
    score = accuracy_score(Y_test, Y_Hat)
    loss = log_loss(Y_test, Y_Hat_prob)

    print('Regressao Linear: ')
    print('Precisão: ' + str(round(score, 4)) + ' Logloss: ' + str(round(loss, 4)))
    print('Tempo de execução:' + str(end - start))

    start = time.time()

    classifier = DecisionTreeClassifier(max_depth=4)

    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    end = time.time()
    Y_Hat_prob = classifier.predict_proba(X_test)

    score = accuracy_score(Y_test, Y_Hat)
    loss = log_loss(Y_test, Y_Hat_prob)

    print('Árvore de Decisão: ')
    print('Precisão: ' + str(round(score, 4)) + ' Logloss: ' + str(round(loss, 4)))
    print('Tempo de execução:' + str(end - start))

    start = time.time()

    classifier = GaussianNB()

    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    end = time.time()
    Y_Hat_prob = classifier.predict_proba(X_test)

    score = accuracy_score(Y_test, Y_Hat)
    loss = log_loss(Y_test, Y_Hat_prob)

    print('Bayes: ')
    print('Precisão: ' + str(round(score, 4)) + ' Logloss: ' + str(round(loss, 4)))
    print('Tempo de execução:' + str(end - start))

    start = time.time()

    classifier = MLPClassifier(hidden_layer_sizes=(25, 25, 25))

    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    end = time.time()
    Y_Hat_prob = classifier.predict_proba(X_test)

    score = accuracy_score(Y_test, Y_Hat)
    loss = log_loss(Y_test, Y_Hat_prob)

    print('Rede Neural: ')
    print('Precisão: ' + str(round(score, 4)) + ' Logloss: ' + str(round(loss, 4)))
    print('Tempo de execução:' + str(end - start))

    start = time.time()

    classifier = neighbors.KNeighborsClassifier()

    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    end = time.time()
    Y_Hat_prob = classifier.predict_proba(X_test)

    score = accuracy_score(Y_test, Y_Hat)
    loss = log_loss(Y_test, Y_Hat_prob)

    print('KNN: ')
    print('Precisão: ' + str(round(score, 4)) + ' Logloss: ' + str(round(loss, 4)))
    print('Tempo de execução:' + str(end - start))

    start = time.time()

    classifier = cluster.KMeans(n_clusters=2)
    classifier.fit_transform(X_train)
    Y_Hat = classifier.predict(X_test)
    end = time.time()
    score = accuracy_score(Y_test, Y_Hat)

    print('Cluster: ')
    print('Precisão: ' + str(round(score, 4)))
    print('Tempo de execução:' + str(end - start))

    start = time.time()

    classifier = linear_model.LogisticRegression()
    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    end = time.time()
    Y_Hat_prob = classifier.predict_proba(X_test)

    score = accuracy_score(Y_test, Y_Hat)
    loss = log_loss(Y_test, Y_Hat_prob)

    print('Regressão Logística: ')
    print('Precisão: ' + str(round(score, 4)) + ' Logloss: ' + str(round(loss, 4)))
    print('Tempo de execução:' + str(end - start))

    start = time.time()

    classifier = LinearDiscriminantAnalysis()
    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    end = time.time()
    Y_Hat_prob = classifier.predict_proba(X_test)

    score = accuracy_score(Y_test, Y_Hat)
    loss = log_loss(Y_test, Y_Hat_prob)

    print('LDA: ')
    print('Precisão: ' + str(round(score, 4)) + ' Logloss: ' + str(round(loss, 4)))
    print('Tempo de execução:' + str(end - start))

    start = time.time()

    classifier = algorithms.GRNN(std=1, verbose=False)
    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    end = time.time()
    Y_Hat = utl.squeeze_inner(Y_Hat)

    Y_Hat_prob = utl.get_probability_round_aproximation(Y_Hat)

    Y_Hat = np.around(Y_Hat)
    score = accuracy_score(Y_test, Y_Hat)
    loss = log_loss(Y_test, Y_Hat_prob)

    print('GRNN: ')
    print('Precisão: ' + str(round(score, 4)) + ' Logloss: ' + str(round(loss, 4)))
    print('Tempo de execução:' + str(end - start))

    start = time.time()

    classifier = Earth()
    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    end = time.time()
    Y_Hat_prob = utl.get_probability_round_aproximation(Y_Hat)
    Y_Hat = np.around(Y_Hat)
    score = accuracy_score(Y_test, Y_Hat)
    loss = log_loss(Y_test, Y_Hat_prob)

    end = time.time()

    print('MARS: ')
    print('Precisão: ' + str(round(score, 4)) + ' Logloss: ' + str(round(loss, 4)))
    print('Tempo de execução:' + str(end - start))

    start = time.time()

    classifier = DBSCAN()
    classifier.fit(X_train)
    Y_Hat = classifier.fit_predict(X_test)
    end = time.time()
    score = accuracy_score(Y_test, Y_Hat)

    print('DBSCAN: ')
    print('Precisão: ' + str(round(score, 4)))
    print('Tempo de execução:' + str(end - start))


# Treina os métodos de classificação e mede o desempenho dos mesmos utilizando o PCA
def classificaPCA(X_train, X_test, Y_train, Y_test, components):
    pca = PCA(n_components=components)
    X_train = pca.fit_transform(X_train)
    X_test = pca.fit_transform(X_test)

    start = time.time()
    classifier = SVC(probability=True)

    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    end = time.time()
    Y_Hat_prob = classifier.predict_proba(X_test)

    score = accuracy_score(Y_test, Y_Hat)
    loss = log_loss(Y_test, Y_Hat_prob)
    print('SVM: ')
    print('Precisão: ' + str(round(score, 4)) + ' Logloss: ' + str(round(loss, 4)))
    print('Tempo de execução:' + str(end - start))

    start = time.time()

    classifier = linear_model.LinearRegression()
    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    end = time.time()
    Y_Hat_prob = utl.get_probability_round_aproximation(Y_Hat)
    Y_Hat = np.around(Y_Hat)
    score = accuracy_score(Y_test, Y_Hat)
    loss = log_loss(Y_test, Y_Hat_prob)

    print('Regressao Linear: ')
    print('Precisão: ' + str(round(score, 4)) + ' Logloss: ' + str(round(loss, 4)))
    print('Tempo de execução:' + str(end - start))

    start = time.time()

    classifier = DecisionTreeClassifier(max_depth=4)

    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    end = time.time()
    Y_Hat_prob = classifier.predict_proba(X_test)

    score = accuracy_score(Y_test, Y_Hat)
    loss = log_loss(Y_test, Y_Hat_prob)

    print('Árvore de Decisão: ')
    print('Precisão: ' + str(round(score, 4)) + ' Logloss: ' + str(round(loss, 4)))
    print('Tempo de execução:' + str(end - start))

    start = time.time()

    classifier = GaussianNB()

    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    end = time.time()
    Y_Hat_prob = classifier.predict_proba(X_test)

    score = accuracy_score(Y_test, Y_Hat)
    loss = log_loss(Y_test, Y_Hat_prob)

    print('Bayes: ')
    print('Precisão: ' + str(round(score, 4)) + ' Logloss: ' + str(round(loss, 4)))
    print('Tempo de execução:' + str(end - start))

    start = time.time()

    classifier = MLPClassifier(hidden_layer_sizes=(25, 25, 25))

    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    end = time.time()
    Y_Hat_prob = classifier.predict_proba(X_test)

    score = accuracy_score(Y_test, Y_Hat)
    loss = log_loss(Y_test, Y_Hat_prob)

    print('Rede Neural: ')
    print('Precisão: ' + str(round(score, 4)) + ' Logloss: ' + str(round(loss, 4)))
    print('Tempo de execução:' + str(end - start))

    start = time.time()

    classifier = neighbors.KNeighborsClassifier()

    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    end = time.time()
    Y_Hat_prob = classifier.predict_proba(X_test)

    score = accuracy_score(Y_test, Y_Hat)
    loss = log_loss(Y_test, Y_Hat_prob)

    print('KNN: ')
    print('Precisão: ' + str(round(score, 4)) + ' Logloss: ' + str(round(loss, 4)))
    print('Tempo de execução:' + str(end - start))

    start = time.time()

    classifier = cluster.KMeans(n_clusters=2)
    classifier.fit_transform(X_train)
    Y_Hat = classifier.predict(X_test)
    end = time.time()
    score = accuracy_score(Y_test, Y_Hat)

    print('Cluster: ')
    print('Precisão: ' + str(round(score, 4)))
    print('Tempo de execução:' + str(end - start))

    start = time.time()

    classifier = linear_model.LogisticRegression()
    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    end = time.time()
    Y_Hat_prob = classifier.predict_proba(X_test)

    score = accuracy_score(Y_test, Y_Hat)
    loss = log_loss(Y_test, Y_Hat_prob)

    print('Regressão Logística: ')
    print('Precisão: ' + str(round(score, 4)) + ' Logloss: ' + str(round(loss, 4)))
    print('Tempo de execução:' + str(end - start))

    start = time.time()

    classifier = LinearDiscriminantAnalysis()
    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    end = time.time()
    Y_Hat_prob = classifier.predict_proba(X_test)

    score = accuracy_score(Y_test, Y_Hat)
    loss = log_loss(Y_test, Y_Hat_prob)

    print('LDA: ')
    print('Precisão: ' + str(round(score, 4)) + ' Logloss: ' + str(round(loss, 4)))
    print('Tempo de execução:' + str(end - start))

    start = time.time()

    classifier = algorithms.GRNN(std=1, verbose=False)
    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    end = time.time()
    Y_Hat = utl.squeeze_inner(Y_Hat)

    Y_Hat_prob = utl.get_probability_round_aproximation(Y_Hat)

    Y_Hat = np.around(Y_Hat)
    score = accuracy_score(Y_test, Y_Hat)
    loss = log_loss(Y_test, Y_Hat_prob)

    print('GRNN: ')
    print('Precisão: ' + str(round(score, 4)) + ' Logloss: ' + str(round(loss, 4)))
    print('Tempo de execução:' + str(end - start))

    start = time.time()

    classifier = Earth()
    classifier.fit(X_train, Y_train)
    Y_Hat = classifier.predict(X_test)
    end = time.time()
    Y_Hat_prob = utl.get_probability_round_aproximation(Y_Hat)
    Y_Hat = np.around(Y_Hat)
    score = accuracy_score(Y_test, Y_Hat)
    loss = log_loss(Y_test, Y_Hat_prob)

    end = time.time()

    print('MARS: ')
    print('Precisão: ' + str(round(score, 4)) + ' Logloss: ' + str(round(loss, 4)))
    print('Tempo de execução:' + str(end - start))

    start = time.time()

    classifier = DBSCAN()
    classifier.fit(X_train)
    Y_Hat = classifier.fit_predict(X_test)
    end = time.time()
    score = accuracy_score(Y_test, Y_Hat)

    print('DBSCAN: ')
    print('Precisão: ' + str(round(score, 4)))
    print('Tempo de execução:' + str(end - start))