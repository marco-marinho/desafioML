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

Durante a análise inicial das variáriaveis identificou-se que as variáveis *capital-gain*, *capital-loss*, *native-country*, 
*race* apresentaram forte viés. Como, em geral, métodos de regressão e previsão se coportam melhor quando as variáveis preditoras
tem distribuições mais próximas da normal, a raiz cúbica foi utilizada como método para a redução do viés. A raiz cubica foi 
escolhida por ser suficiente para a redução do vies e por ser uma transformação menos forte que a transformação logaritimica.

### Processo de normalização dos dados:

As variáveis preditoras foram normalizadas de forma a possuir média 0 e desvio padrão 1. Essa escolha deve-se novamente ao fato de
que a maioria dos métodos de regressão possuirem desempenho melhor quando as variáveis preditoras tem distribuiçes proximas da
distribuição normal. A transformação foi feita subtraindo-se de cada entrada a média do conjunto e divido-a pelo desvio padrão
do conjunto.

### Modelos de previsão e desempenho:

Os modelos de previsão utilizados no estudos foram: support vector machines (SVM), regressão linear, árvores de decisão,
naive bayes, redes neurais, k-nearest neighbors (KNN), clustering, regressão logística, linear discriminant analysis (LDA),
generalized regression neural networkds (GRNN) e multivariate adaptive regression splines (MARS). 

A decomposição em componentes principais (PCA) também foi estudada como forma de melhorar o desempenho computacional. Utilizando
o PCA para reduzir a dimensionalidade do problema foi possível reduzir o tempo de processamento em aproximadamente 40% com uma
redução de precisão em torno de 3%.

### Modelos supervisionados utilizados:

Os seguintes modelos supervisionados foram utilizados: upport vector machines (SVM), regressão linear, árvores de decisão,
naive bayes, redes neurais, k-nearest neighbors (KNN), regressão logística, linear discriminant analysis (LDA),
generalized regression neural networkds (GRNN) e multivariate adaptive regression splines (MARS).

O único modelo não supervisionado utilizado foi o modelo de clustering. Além disso, o PCA utilizado para decomposição de dados
pode ser também considerado como não supervisionado, visto que a separação em componentes ortogonais é feita de forma cega.

### Treino e evolução:

Para o treino dos classificadores foram utilizados os primeiros 20% do conjunto de dados. Os 80% restantes foram utilizados para a validação dos classificadores.

### Desempenho do modelo e revalidação log loss

A seguir são apresentados o desepenho de todos os métodos e a revalidação log loss para os métodos que apresentam a probabilidade de classificação bem definida. Os resultados são apresentados tanto para o caso de aplicação do PCA quanto para
o caso dos dados com a dimensionalidade original. Além disso, é a presentado o tempo de processamento total para todos os métodos utilizado ou não PCA de forma a demonstrar o ganho computacional obtido.

Desempenho para dados com dimensionalidade original:

| Método        | Desempenho           | Logloss  |
| ------------- |:-------------:| -----:|
| SVM   | 0.8582 | 0.3451 |
| Regressão Linear | 0.8455 | N/A |
| Árvore de decisão | 0.8302  | 0.3571 |
| Bayes | 0.8094 | 0.6151 |
| Rede Neural | 0.8473 | 0.3199 |
| KNN | 0.8356 | 1.148 |
| Cluster | 0.6658 | N/A |
| Regressão Logística | 0.8455 | 0.3287 |
| LDA | 0.8464 | 0.3343 |
| GRNN | 0.8546 | N/A |
| MARS | 0.8464 | N/A |

Tempo de execução:5.363988637924194


Desempenho utilizando PCA:

| Método        | Desempenho           | Logloss  |
| ------------- |:-------------:| -----:|
| SVM   | 0.8410 | 0.4017 |
| Regressão Linear | 0.8302 | N/A |
| Árvore de decisão | 0.8302  | 0.4306 |
| Bayes | 0.8283 | 0.3670 |
| Rede Neural | 0.8365 | 0.3497 |
| KNN | 0.8202 | 1.14845 |
| Cluster | 0.6811 | N/A |
| Regressão Logística | 0.8229 | 0.3630 |
| LDA | 0.8293 | 0.3634 |
| GRNN | 0.8428 | N/A |
| MARS | 0.8184 | N/A |

Tempo de execução:3.3925909996032715
