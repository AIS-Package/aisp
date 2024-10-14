## Português

A **seleção negativa** é o processo em que o sistema imunológico faz a maturação das células-T conhecidas também por linfócitos-T, no qual tornam-as aptas na detecção dos não-próprios. Assim, o Algoritmo de seleção negativa (NSA), utilizam-se de hiperesferas simbolizando os detectores em um espaço de dados N-dimensional. [[1]](#ref1)

---

# RNSA (Algoritmo de Seleção Negativa de Valor Real)

Esta classe estende a classe [**Base**](../advanced-guides/BaseNegativeSelection.md#português).

## Construtor RNSA:

A classe ``RNSA`` tem a finalidade de classificação e identificação de anomalias através do método self e not self . 

**Attributes:**
* *N* (``int``): Quantidade de detectores. Defaults to ``100``.
* *r* (``float``): Raio do detector. Defaults to ``0.05``.
* *k* (``int``): Quantidade de vizinhos próximos dos detectores gerados aleatoriamente para efetuar o cálculo da média da distância. Defaults to ``1``.
* *metric* (``str``): Forma para se calcular a distância entre o detector e a amostra: 
    
    * ``'euclidiana'`` ➜ O cálculo da distância dá-se pela expressão:  √( (X₁ – X₂)² + (Y₁ – Y₂)² + ... + (Yn – Yn)²).
    * ``'minkowski'``  ➜ O cálculo da distância dá-se pela expressão: ( |X₁ – Y₁|p + |X₂ – Y₂|p + ... + |Xn – Yn|p) ¹/ₚ , Neste projeto ``p == 2``.
    * ``'manhattan'``  ➜ O cálculo da distância dá-se pela expressão:  ( |X₁ – X₂| + |Y₁ – Y₂| + ... + |Yn – Yn₂|).

    Defaults to ``'euclidean'``.

* *max_discards* (``int``): Este parâmetro indica o número máximo de descartes de detectores em sequência, que tem como objetivo evitar um 
possível loop infinito caso seja definido um raio que não seja possível gerar detectores do não-próprio.

* *seed* (``int``): Semente para a geração randômica dos valores nos detectores. Defaults to ``None``.
* *algorithm* (``str``), Definir a versão do algoritmo:

    * ``'default-NSA'``: Algoritmo padrão com raio fixo.
    * ``'V-detector'``: Este algoritmo é baseado no artigo "[Real-Valued Negative Selection Algorithm with Variable-Sized Detectors](https://doi.org/10.1007/978-3-540-24854-5_30)", de autoria de Ji, Z., Dasgupta, D. (2004), e utiliza um raio variável para a detecção de anomalias em espaços de características.  

    Defaults to ``'default-NSA'``.

* *r_s* (``float``): O valor de ``rₛ`` é o raio das amostras próprias da matriz ``X``.
* ``**kwargs``:
    - *non_self_label* (``str``): Esta variável armazena o rótulo que será atribuído quando os dados possuírem 
    apenas uma classe de saída, e a amostra for classificada como não pertencente a essa classe. Defaults to ``'non-self'``.
    - *cell_bounds* (``bool``):  Se definido como ``True``, esta opção limita a geração dos detectores ao espaço do plano 
    compreendido entre 0 e 1. Isso significa que qualquer detector cujo raio ultrapasse esse limite é descartado, 
    e esta variável é usada exclusivamente no algoritmo ``V-detector``.

**Outras variáveis iniciadas:**

* *detectors* (``dict``): Esta variável armazena uma lista de detectores por classe.

* *classes* (``npt.NDArray``): lista de classes de saída.

---

## Função fit(...)

A função ``fit(...)`` gera os detectores para os não próprios com relação às amostras:

```python
def fit(self, X: npt.NDArray, y: npt.NDArray):
```
Nela é realizado o treinamento de acordo com ``X`` e ``y``, usando o método de seleção negativa(``NegativeSelect``).

**Os parâmetros de entrada são:**
* ``X``: array com as características das amostras com **N** amostras (linhas) e **N** características  (colunas), normalizados para valores entre [0, 1]. 
* ``y``: array com as classes de saídas disposto em **N** amostras que são relacionadas ao ``X``.
* ``verbose``: boolean com valor default ``True``, determina se o feedback da geração dos detectores será imprimido.

*Retorna a instância da classe.*

---

### Função predict(...)

A função ``predict(...)`` realiza a previsão das classes utilizando os detectores gerados:

```python
def predict(self, X: npt.NDArray) -> npt.NDArray:
```

**O parâmetro de entrada:**
 
* ``X``: array  com as características para a previsão, com **N** amostras (Linhas) e **N** colunas.

**Retorna:** 
* ``C``: Um array de previsão com as classes de saída para as características informadas. 
* ``None``: se não houver detectores.

---

### Função score(...):

A função "score(...)" calcula a precisão do modelo treinado por meio da realização de previsões e do cálculo da acurácia.

```python
def score(self, X: npt.NDArray, y: list) -> float:
```

retorna a acurácia, do tipo ``float``.

---

## Métodos privados

---

### Função __checks_valid_detector(...):

A função ``def __checks_valid_detector(...)`` verifica se o detector possui raio ``r`` válido para o não-próprio da classe:

```python
def __checks_valid_detector(self, X: npt.NDArray, vector_x: npt.NDArray, samplesIndexClass: npt.NDArray) -> bool:
```

**Os parâmetros de entrada são:**
* ``X``: array com as características das amostras com **N** amostras (linhas) e **N** características  (colunas), normalizados para valores entre [0, 1].

* ``vector_x``: Detector candidato gerado aleatoriamente. 

* ``samplesIndexClass``: Array com os indexes de uma classe.

**Retorna:** Verdadeiro (``True``) para os detectores que não possuam amostras em seu interior ou falso (``False``) se possuir.

---

### Função __compare_KnearestNeighbors_List(...):

A função ``def __compare_KnearestNeighbors_List(...)`` compara a distância dos k-vizinhos mais próximo, para isso se a distância da nova amostra for menor, substitui ``k-1`` e ordena em ordem crescente:

```python
def __compare_KnearestNeighbors_List(self, knn: npt.NDArray, distance: float) -> npt.NDArray:
```

**Retorna:** uma lista com as distâncias dos k-vizinhos mais próximo.

---

### Função __compare_sample_to_detectors(...):

Função para comparar uma amostra com os detectores, verificando se a amostra é própria.


Nesta função, quando possui ambiguidade de classes, retorna a classe que possuir a média de distância maior entre os detectores.

**Os parâmetros de entrada são:**
* line: vetor com N-características

**Retorna:** A classe prevista com os detectores ou None se a amostra não se qualificar a nenhuma classe.
       
---

### Função __detector_is_valid_to_Vdetector(...):

Verifique se a distância entre o detector e as amostras, descontando o raio das amostras, é maior do que o raio mínimo.

```python
def __detector_is_valid_to_Vdetector(self, distance, vector_x):
```

**Os parâmetros de entrada são:**

* distance (``float``): distância mínima calculada entre todas as amostras.
* vector_x (``numpy.ndarray``): vetor x candidato do detector gerado aleatoriamente, com valores entre 0 e 1.

**Retorna:**

* ``False``: caso o raio calculado seja menor do que a distância mínima ou ultrapasse a borda do espaço, caso essa opção esteja habilitada.
* ``True`` e a distância menos o raio das amostras, caso o raio seja válido.

---

### Função __distance(...):

A função ``def __distance(...)`` calcula a distância entre dois pontos utilizando a técnica definida em ``metric``, no qual são: ``'euclidiana', 'minkowski', ou 'manhattan'``

```python
def __distance(self, u: npt.NDArray, v: npt.NDArray):
```

Os parâmetros de entrada são NDArrays: ``u`` e ``v``, com as coordenadas para os pontos.

Retorna a distancia (``double``) entre os dois pontos.

---

### Função __slice_index_list_by_class(...):

A função ``__slice_index_list_by_class(...)``, separa os índices das linhas conforme a classe de saída, para percorrer o array de amostra, apenas nas posições que a saída for a classe que está sendo treinada:

```python
def __slice_index_list_by_class(self, y: npt.NDArray) -> dict:
```

Retorna um dicionario com as classes como chave e os índices em ``X`` das amostras.

---

# BNSA (Algoritmo de Seleção Negativa Binária).

Esta classe estende a classe [**Base**](../advanced-guides/BaseNegativeSelection.md#português).

## Construtor BNSA:

A classe ``BNSA`` tem a finalidade de classificação e identificação de anomalias através do método self e not self . 

**Attributes:**
* *N* (``int``): Quantidade de detectores. Defaults to ``100``.
* *aff_thresh* (``float``): A variável representa a porcentagem de não similaridade entre a célula T e as amostras próprias. O valor padrão é de 10% (0.1), enquanto que o valor de 1.0 representa 100% de não similaridade.
* *max_discards* (``int``): Este parâmetro indica o número máximo de descartes de detectores em sequência, que tem como objetivo evitar um 
possível loop infinito caso seja definido um raio que não seja possível gerar detectores do não-próprio. Defaults to ``100``.
* *seed* (``int``): Semente para a geração randômica dos valores nos detectores. Defaults to ``None``.
* no_label_sample_selection (``str``): Método para a seleção de rótulos para amostras designadas como não pertencentes por todos os detectores não pertencentes. **Tipos de métodos disponíveis:**
    - (``max_average_difference``): Seleciona a classe com a maior diferença média entre os detectores.
    - (``max_nearest_difference``): Seleciona a classe com a maior diferença entre o detector mais próximo e mais distante da amostra.

**Outras variáveis iniciadas:**

* *detectors* (``dict``): Esta variável armazena uma lista de detectores por classe.

* *classes* (``npt.NDArray``): lista de classes de saída.

### Função fit(...)

A função ``fit(...)`` gera os detectores para os não próprios com relação às amostras:

```python
def fit(self, X: npt.NDArray, y: npt.NDArray):
```
Nela é realizado o treinamento de acordo com ``X`` e ``y``, usando o método de seleção negativa(``NegativeSelect``).

**Os parâmetros de entrada são:**
* ``X``: array com as características das amostras com **N** amostras (linhas) e **N** características  (colunas), normalizados para valores entre [0, 1]. 
* ``y``: array com as classes de saídas disposto em **N** amostras que são relacionadas ao ``X``.
* ``verbose``: boolean com valor default ``True``, determina se o feedback da geração dos detectores será imprimido.

*Retorna a instância da classe.*

---

### Função predict(...)

A função ``predict(...)`` realiza a previsão das classes utilizando os detectores gerados:

```python
def predict(self, X: npt.NDArray) -> npt.NDArray:
```

**O parâmetro de entrada:**
 
* ``X``: array  com as características para a previsão, com **N** amostras (Linhas) e **N** colunas.

**Retorna:** 
* ``C``: Um array de previsão com as classes de saída para as características informadas. 
* ``None``: se não houver detectores.

---

### Função score(...):

A função "score(...)" calcula a precisão do modelo treinado por meio da realização de previsões e do cálculo da acurácia.

```python
def score(self, X: npt.NDArray, y: list) -> float:
```

retorna a acurácia, do tipo ``float``.

---

## Métodos privados

---

### Função __slice_index_list_by_class(...):

A função ``__slice_index_list_by_class(...)``, separa os índices das linhas conforme a classe de saída, para percorrer o array de amostra, apenas nas posições que a saída for a classe que está sendo treinada:

```python
def __slice_index_list_by_class(self, y: npt.NDArray) -> dict:
```

Retorna um dicionario com as classes como chave e os índices em ``X`` das amostras.



## English

**Negative selection** is the process in which the immune system maturates T-cells, also known as T-lymphocytes, which make them capable of detecting non-self. Thus, the Negative Selection Algorithm (NSA) uses hyperspheres symbolizing the detectors in an N-dimensional data space. [[1]](#1)

---


# RNSA (Real-Valued Negative Selection Algorithm)

This class extends the [**Base**](../advanced-guides/BaseNegativeSelection.md#english) class.

## Constructor RNSA:

The ``RNSA`` class has the purpose of classifying and identifying anomalies through the self and not self methods.


**Attributes:**

* *N* (``int``): Number of detectors. Defaults to ``100``.
* *r* (``float``): Radius of the detector. Defaults to ``0.05``.
* *k* (``int``): Number of neighbors near the randomly generated detectors to perform the distance average calculation. Defaults to ``1``.
* *metric* (``str``): Way to calculate the distance between the detector and the sample:

    * ``'Euclidean'`` ➜ The calculation of the distance is given by the expression: √( (X₁ – X₂)² + (Y₁ – Y₂)² + ... + (Yn – Yn)²).

    * ``'minkowski'`` ➜ The calculation of the distance is given by the expression: ( |X₁ – Y₁|p + |X₂ – Y₂|p + ... |Xn – Yn|p) ¹/ₚ , In this project ``p == 2``.
    * ``'manhattan'`` ➜ The calculation of the distance is given by the expression: ( |X₁ – X₂| + |Y₁ – Y₂| + ...+ |Yn – Yn₂|) .

    Defaults to ``'euclidean'``.


* *max_discards* (``int``): This parameter indicates the maximum number of consecutive detector discards, aimed at preventing a possible infinite loop in case a radius is defined that cannot generate non-self detectors.
* *seed* (``int``): Seed for the random generation of values in the detectors. Defaults to ``None``.

* *algorithm* (``str``), Set the algorithm version:

    * ``'default-NSA'``: Default algorithm with fixed radius.
    * ``'V-detector'``: This algorithm is based on the article "[Real-Valued Negative Selection Algorithm with Variable-Sized Detectors](https://doi.org/10.1007/978-3-540-24854-5_30)", by Ji, Z., Dasgupta, D. (2004), and uses a variable radius for anomaly detection in feature spaces.

    Defaults to ``'default-NSA'``.

* *r_s* (``float``): rₛ Radius of the ``X`` own samples.
* ``**kwargs``:
    - *non_self_label* (``str``): This variable stores the label that will be assigned when the data has only one 
    output class, and the sample is classified as not belonging to that class. Defaults to ``'non-self'``.
    - *cell_bounds* (``bool``): If set to ``True``, this option limits the generation of detectors to the space within 
    the plane between 0 and 1. This means that any detector whose radius exceeds this limit is discarded, 
    this variable is only used in the ``V-detector`` algorithm. Defaults to ``False``.


**Other variables initiated:**

* *detectors* (``dict``): This variable stores a list of detectors by class.

* *classes* (``npt.NDArray``): list of output classes.

---

### Function fit(...)

The ``fit(...)`` function generates the detectors for non-fits with respect to the samples:

```python
def fit(self, X: npt.NDArray, y: npt.NDArray):
```

In it, training is performed according to ``X`` and ``y``, using the negative selection method(``NegativeSelect``).

The input parameters are: 
* ``X``: array with the characteristics of the samples with **N** samples (rows) and **N** characteristics (columns). 

* ``y``: array with the output classes arranged in **N** samples that are related to ``X``.

* ``verbose``: boolean with default value ``True``, determines if the feedback from the detector generation will be printed.

*Returns the instance of the class.*

---

### Function predict(...)

The ``predict(...)`` function performs class prediction using the generated detectors:

```python
def predict(self, X: npt.NDArray) -> npt.NDArray:
```

**The input parameter is:** 
* ``X``: array with the characteristics for the prediction, with **N** samples (Rows) and **N** columns.

**Returns:** 
* ``C``: prediction array, with the output classes for the given characteristics.
* ``None``: if there are no detectors.

---

### Function score(...):

The function ``score(...)`` calculates the accuracy of the trained model by making predictions and computing accuracy.

```python
def score(self, X: npt.NDArray, y: list) -> float:
```

It returns the accuracy as a float type.

---

## Private Methods

---

### Function __checks_valid_detector(...):

The ``def __checks_valid_detector(...)`` function checks if the detector has a valid ``r`` radius for the non-self of the class:

```python
def __checks_valid_detector(self, X: npt.NDArray, vector_x: npt.NDArray, samplesIndexClass: npt.NDArray) -> bool:
```

**The input parameters are:**
* ``X``: array with sample characteristics with **N** samples (rows) and **N** characteristics (columns), normalized to values between [0, 1].

* ``vector_x``: Randomly generated candidate detector.

* ``samplesIndexClass``: Array with the indexes of a class.
 
**Returns:** ``True`` for detectors that do not have samples inside or ``False`` if they do.

---

### Function __compare_KnearestNeighbors_List(...):

The ``def __compare_KnearestNeighbors_List(...)`` function compares the distance of the k-nearest neighbors, so if the distance of the new sample is smaller, replaces ``k-1`` and sorts in ascending order:

```python
def __compare_KnearestNeighbors_List(self, knn: npt.NDArray, distance: float) -> npt.NDArray:
```

Returns a list of k-nearest neighbor distances.

---

### Function __compare_sample_to_detectors(...):

Function to compare a sample with the detectors, verifying if the sample is proper.
In this function, when there is class ambiguity, it returns the class that has the greatest average distance between the detectors.

```python
def __compare_sample_to_detectors(self, line):
```
**The input parameters are:**
* line: vector with N-features
 
**Returns:** The predicted class with the detectors or None if the sample does not qualify for any class.

---

### Function __detector_is_valid_to_Vdetector(...):

Check if the distance between the detector and the samples, minus the radius of the samples, is greater than the minimum radius.

```python
def __detector_is_valid_to_Vdetector(self, distance, vector_x):
```
 
**The input parameters are:**

* distance (``float``): minimum distance calculated between all samples.
* vector_x (``numpy.ndarray``): randomly generated candidate detector vector x with values between 0 and 1.

**Returns:** 
* ``False``: if the calculated radius is smaller than the minimum distance or exceeds the edge of the space, if this option is enabled.
* ``True`` and the distance minus the radius of the samples, if the radius is valid.`

---

### Function __distance(...):

The function ``def __distance(...)`` calculates the distance between two points using the technique defined in ``metric``, which are: ``'euclidean', 'norm_euclidean', or 'manhattan'``

```python
def __distance(self, u: npt.NDArray, v: npt.NDArray):
```

The input parameters are ``u`` and ``v`` NDArrays, with the coordinates for the points.

**Returns:** the distance (``double``) between the two points.

---

### Function __slice_index_list_by_class(...):

The function ``__slice_index_list_by_class(...)``, separates the indices of the lines according to the output class, to go through the sample array, only in the positions that the output is the class that is being trained:

```python
def __slice_index_list_by_class(self, y: npt.NDArray) -> dict:
```

Returns a dictionary with the classes as key and the indices in ``X`` of the samples.

---

# BNSA (Binary Negative Selection Algorithm)

This class extends the [**Base**](../advanced-guides/BaseNegativeSelection.md#english) class.

## Constructor RNSA:

The ``BNSA`` (Binary Negative Selection Algorithm) class has the purpose of classifying and identifying anomalies through the self and not self methods.

**Attributes:**

* *N* (``int``): Number of detectors. Defaults to ``100``.
* *aff_thresh* (``float``): The variable represents the percentage of dissimilarity between the T cell and the own samples. The default value is 10% (0.1), while a value of 1.0 represents 100% dissimilarity.
* *max_discards* (``int``): This parameter indicates the maximum number of detector discards in sequence, which aims to avoid a
possible infinite loop if a radius is defined that it is not possible to generate non-self detectors. Defaults to ``100``.
* *seed* (``int``): Seed for the random generation of values in the detectors. Defaults to ``None``.
* no_label_sample_selection (``str``): Method for selecting labels for samples designated as non-members by all non-member detectors. **Available method types:**
    - (``max_average_difference``): Selects the class with the highest average difference among the detectors.
    - (``max_nearest_difference``): Selects the class with the highest difference between the nearest and farthest detector from the sample.

**Other variables initiated:**

* *detectors* (``dict``): This variable stores a list of detectors by class.

* *classes* (``npt.NDArray``): list of output classes.



### Function fit(...)

The ``fit(...)`` function generates the detectors for non-fits with respect to the samples:

```python
def fit(self, X: npt.NDArray, y: npt.NDArray):
```

In it, training is performed according to ``X`` and ``y``, using the negative selection method(``NegativeSelect``).

**The input parameters are:** 
* ``X``: array with the characteristics of the samples with **N** samples (rows) and **N** characteristics (columns). 

* ``y``: array with the output classes arranged in **N** samples that are related to ``X``.

* ``verbose``: boolean with default value ``True``, determines if the feedback from the detector generation will be printed.

*Returns the instance of the class.*

---

### Function predict(...)

The ``predict(...)`` function performs class prediction using the generated detectors:

```python
def predict(self, X: npt.NDArray) -> npt.NDArray:
```

**The input parameter is:** 
* ``X``: array with the characteristics for the prediction, with **N** samples (Rows) and **N** columns.

**Returns:** 
* ``C``: prediction array, with the output classes for the given characteristics.
* ``None``: if there are no detectors.

---

### Function score(...):

The function ``score(...)`` calculates the accuracy of the trained model by making predictions and computing accuracy.

```python
def score(self, X: npt.NDArray, y: list) -> float:
```

It returns the accuracy as a float type.

---

## Private Methods

---

### Function __slice_index_list_by_class(...):

The function ``__slice_index_list_by_class(...)``, separates the indices of the lines according to the output class, to go through the sample array, only in the positions that the output is the class that is being trained:

```python
def __slice_index_list_by_class(self, y: npt.NDArray) -> dict:
```

Returns a dictionary with the classes as key and the indices in ``X`` of the samples.

---

# References

<br id='ref1'/>

> 1. BRABAZON, Anthony; O’NEILL, Michael; MCGARRAGHY, Seán. Natural Computing Algorithms. [S. l.]: Springer Berlin Heidelberg, 2015. DOI 10.1007/978-3-662-43631-8. Disponível em: http://dx.doi.org/10.1007/978-3-662-43631-8.

<br id='ref2'/>