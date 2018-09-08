import pandas as pan
import numpy as np
import util as util
import seaborn as sbrn
import matplotlib.pyplot as pyplt
from sklearn.model_selection import train_test_split
from classifiers import classifica, classificaPCA

pan.options.mode.chained_assignment = None

data = pan.read_csv('data.csv')

# Remove campos com valores nan
data = data.dropna()

# Casta os campos string para numericos de acordo com a variavel income
data = util.cast_to_numeric(data)

# Corrige o vies de alguns campos
data = util.correct_skewness(data, ['capital-gain', 'capital-loss', 'native-country', 'race'])

# Transforma variaveis correlacionadas em uma só
data['marital-relation'] = data['marital-status'] * data['relationship']

# Seleciona quais são os campos numéricos, remove o campo income e normaliza os dados
numeric = data.select_dtypes(include=[np.number]).columns.tolist()
numeric.remove('income')
data[numeric] = util.normalize_data(data[numeric])

# Plota a matriz de correlação para facilitar a análise dos dados
corrmat = data.corr()
f, ax = pyplt.subplots(figsize=(12, 9))
sbrn.heatmap(corrmat, annot=True, vmax=.8, square=True, cmap="RdBu_r")
pyplt.show()

# Separa os dados em um conjunto de testes e um de treino
X = data[['marital-relation', 'workclass', 'occupation', 'age', 'education-num', 'hours-per-week', 'capital-gain',
          'sex', 'native-country', 'capital-loss']]
Y = data['income']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

print('Desempenho sem PCA:')

# Treina e testa a performance dos métodos sem usar o PCA
classifica(X_train, X_test, Y_train, Y_test)

print(' ')
print('Desempenho com PCA:')

# Treina e testa a performance dos métodos usando o PCA
classificaPCA(X_train, X_test, Y_train, Y_test, 2)
