### Variáveis exploradas:

Todas as variáveis disponibilizadas no dataset foram exploradas inicialmente. A exploração de todas as variáveis buscou identificar correlação entre as variáveis preditoras e a variável de resposta. 

### Novas variáveis:

Durante a análise inicial de todas as variáveis foi identificada forte correlação entre as preditoras *marital-status* e *relationship*. 
Portanto, para evitar que houvesse imprecisão causada por correlação entre as preditoras, essas duas variáveis foram transformadas em uma só variável *marital-relation*.

### Preparação do dataset:

O processo de preparação inicial consistiu na remoção de tripas de dados que apresentassem má formação na variável *income* 
e a remoção de espaços nas variáveis que consistiam de strings de caracteres. Em seguida, as variáveis representadas por 
strings foram substituídas por variáveis numéricas. A variável de resposta foi substituída por uma variável binária. 
Os índices numéricos utilizados para a substituição foram escolhidos de acordos com a relação que as variáveis a serem 
substituídas possuíam com a variável de resposta. Isto é, a ordem dos índices dados seguiu a ordem das variáveis que
tivessem o maior número de variáveis de resposta iguais a 1. Com isso, buscou-se facilitar a identificação do nível de 
correlação entre as preditoras e a variável de resposta.

### Transformação de recursos enviesados:

Durante a análise inicial das variaríeis identificou-se que as variáveis *capital-gain*, *capital-loss*, *native-country* e 
*race* apresentaram forte viés. Como, em geral, métodos de regressão e previsão se comportam melhor quando as variáveis preditoras
tem distribuições mais próximas da normal, a raiz cúbica foi utilizada como método para a redução do viés. A raiz cúbica foi 
escolhida por ser suficiente para a redução do viés e por ser uma transformação menos forte que a transformação logarítmica.

### Processo de normalização dos dados:

As variáveis preditoras foram normalizadas de forma a possuir média 0 e desvio padrão 1. Essa escolha deve-se novamente ao fato de
que a maioria dos métodos de regressão possuem desempenho melhor quando as variáveis preditoras tem distribuições próximas da
distribuição normal. A transformação foi feita subtraindo-se de cada entrada a média do conjunto e dividindo-a pelo desvio padrão
do conjunto.

### Modelos de previsão e desempenho:

Os modelos de previsão utilizados no estudo foram: support vector machines (SVM), regressão linear, árvores de decisão,
naive bayes, redes neurais, k-nearest neighbors (KNN), clustering, regressão logística, linear discriminant analysis (LDA),
generalized regression neural networks (GRNN) e multivariate adaptive regression splines (MARS). 

A decomposição em componentes principais (PCA) também foi estudada como forma de melhorar o desempenho computacional. Utilizando
o PCA para reduzir a dimensionalidade do problema foi possível reduzir o tempo de processamento em aproximadamente 40% com uma
redução de precisão em torno de 3%.

### Modelos supervisionados utilizados:

Os seguintes modelos supervisionados foram utilizados: support vector machines (SVM), regressão linear, árvores de decisão,
naive bayes, redes neurais, k-nearest neighbors (KNN), regressão logística, linear discriminant analysis (LDA),
generalized regression neural networks (GRNN) e multivariate adaptive regression splines (MARS).

O único modelo não supervisionado utilizado foi o modelo de clustering. Além disso, o PCA utilizado para decomposição de dados
pode ser também considerado como não supervisionado visto que a separação em componentes ortogonais é feita de forma cega.

### Treino e evolução:

Para o treino dos classificadores foram utilizados os primeiros 20% do conjunto de dados. Os 80% restantes foram utilizados para a validação dos classificadores.

### Desempenho do modelo e revalidação log loss

A seguir são apresentados o desempenho de todos os métodos e a revalidação log loss para os métodos que apresentam a probabilidade de classificação bem definida. Os resultados são apresentados tanto para o caso de aplicação do PCA quanto para
o caso dos dados com a dimensionalidade original. Além disso, é apresentado o tempo de processamento total para todos os métodos utilizando ou não PCA de forma a demonstrar o ganho computacional obtido.

Desempenho para dados com dimensionalidade original:

| Método        | Desempenho           | Logloss  | Tempo de execução|
| ------------- |:-------------:|:-----:| ---------:|
| SVM   | 85.91% | 0.3422 |  2.66152 s  |
| Regressão Linear | 85.37% | 0.3643 | 0.00187 s|
| Árvore de decisão | 83.02%  | 0.3563 | 0.00548 s|
| Bayes | 82.11% | 0.7128 | 0.00231 s|
| Rede Neural | 84.55% | 0.3281 | 2.21614 s|
| KNN | 82.75% | 1.118 | 0.12013 s |
| Cluster | 66.58% | N/A | 0.12633 s|
| Regressão Logística | 85.00% | 0.3247 | 0.01146 s|
| LDA | 84.82% | 0.3298 | 0.00926 s|
| GRNN | 85.46% | 0.3309 | 0.40414 s |
| MARS | 85.37% | 0.3640 | 0.19407 s |
| DBSCAN | 2.08% | N/A | 0.44142 s|

Desempenho utilizando PCA:

| Método        | Desempenho           | Logloss  | Tempo de execução|
| ------------- |:-------------:|:-----:| ---------:|
| SVM   | 84.37% | 0.3999 |  2.33524 s  |
| Regressão Linear | 82.93% | 0.3865 | 0.00093 s|
| Árvore de decisão | 81.93%  | 0.3589 | 0.00509 s|
| Bayes | 83.02% | 0.3644 | 0.00155 s|
| Rede Neural | 83.92% | 0.3505 | 0.32451 s|
| KNN | 81.84% | 1.5664 | 0.00695 s |
| Cluster | 68.38% | N/A | 0.07054 s|
| Regressão Logística | 83.11% | 0.3613 | 0.00698 s|
| LDA | 83.38% | 0.3615 | 0.00268 s|
| GRNN | 84.01% | 0.3534 | 0.23445 s |
| MARS | 83.20% | 0.3579 | 0.08382 s |
| DBSCAN | 74.80% | N/A | 0.10417 s|
