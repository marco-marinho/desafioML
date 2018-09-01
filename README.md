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
correlação entre as proditoras e a variável de resposta.

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
o PCA para reduzir a dimensionalidade do problema foi possível reduzir o tempo de processamento em aproximadamente 50% com uma
redução de precisão de aproximadamente 5%.

### Modelos supervisionados utilizados:

Os seguintes modelos supervisionados foram utilizados: upport vector machines (SVM), regressão linear, árvores de decisão,
naive bayes, redes neurais, k-nearest neighbors (KNN), regressão logística, linear discriminant analysis (LDA),
generalized regression neural networkds (GRNN) e multivariate adaptive regression splines (MARS).

O único modelo não supervisionado utilizado foi o modelo de clustering. Além disso, o PCA utilizado para decomposição de dados
pode ser também considerado como não supervisionado, visto que a separação em componentes ortogonais é feita de forma cega.

### Treino e evolução:

Para o treino dos classificadores foram utilizados os primeiros 20% do conjunto de dados.


