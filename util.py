import numpy as np

#Define os indices de substituição numéricos com base na média 
#do parametro referencia para cada um, facilitando a identificação
#da correlação entre o indice e a referência.
def getOrderedRemapDict(data,target,reference):
    ordered_data = data.groupby(target)[reference].mean().sort_values(ascending=True)
    dictRemap={}
    index=0
    for aux in ordered_data.to_dict():
        dictRemap[aux] = index
        index = index+1
    return dictRemap

#Normaliza os dados para média zero e desvio padrão 1
def normalizeData(data):
    for column in data:
        data[column]=(data[column]-data[column].mean())/data[column].std()
    return data

#Limpa os dados e aplica os indices calculados com getOrderedRemapDict()
def castToNumeric(data):
    data['occupation'] = data['occupation'].str.strip()
    data['income'] = data['income'].str.strip()
    data['income'] = data['income'].map({'<=50K': 0, '>50K': 1}).astype(int)
    data['sex'] = data['sex'].str.strip()
    data['race'] = data['race'].str.strip()
    data['marital-status'] = data['marital-status'].str.strip()
    data['relationship'] = data['relationship'].map(getOrderedRemapDict(data,'relationship','income')).astype(int)
    data['native-country'] = data['native-country'].map(getOrderedRemapDict(data,'native-country','income')).astype(int)
    data['workclass'] = data['workclass'].map(getOrderedRemapDict(data,'workclass','income')).astype(int)
    data['occupation'] = data['occupation'].map(getOrderedRemapDict(data,'occupation','income')).astype(int)
    data['sex'] = data['sex'].map(getOrderedRemapDict(data,'sex','income')).astype(int)
    data['race'] = data['race'].map(getOrderedRemapDict(data,'race','income')).astype(int)
    data['marital-status'] = data['marital-status'].map(getOrderedRemapDict(data,'marital-status','income')).astype(int)
    return data

#Corrige o skew das colunas de dados selecionadas usando a raiz cubica.
def correctSkewness(data, columns):
    for column in columns:
        data[column] = np.cbrt(data[column])
    return data