import pandas as pan
import matplotlib.pyplot as pyplt
import seaborn as sbrn
import numpy as np
import util as util
from sklearn.model_selection import train_test_split
from classifiers import classifica, classificaPCA
import warnings
warnings.filterwarnings("ignore")

data = pan.read_csv('data.csv')
data = data.dropna()
data = util.castToNumeric(data)

#print(data.skew())

data = util.correctSkewness(data,['capital-gain', 'capital-loss','native-country', 'race'])

data['marital-relation'] = data['marital-status']*data['relationship']

numeric = data.select_dtypes(include=[np.number]).columns.tolist()
numeric.remove('income')
data[numeric] = util.normalizeData(data[numeric])

#print(data.std())

corrmat = data.corr()
f, ax = pyplt.subplots(figsize=(12, 9))
sbrn.heatmap(corrmat, annot=True, vmax=.8, square=True, cmap="RdBu_r")
pyplt.show()


X = data[['marital-relation','workclass', 'occupation', 'age', 'education-num', 'hours-per-week','capital-gain', 'sex', 'native-country']]
Y = data['income']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

classifica(X_train,X_test,Y_train,Y_test)

print("  ")

X = data[['marital-relation','workclass', 'occupation', 'age', 'education-num', 'hours-per-week','capital-gain', 'sex', 'native-country']]

classificaPCA(X_train,X_test,Y_train,Y_test,2)