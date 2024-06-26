import numpy as np
import numpy.typing as npt
from tqdm import tqdm
from typing import Dict, Literal, Optional, Union
from collections import namedtuple
from scipy.spatial.distance import cdist

from ._base import Base


class RNSA(Base):
    """
    The ``RNSA`` (Real-Valued Negative Selection Algorithm) class is for classification and \
    identification purposes. of anomalies through the self and not self method.

    Attributes:
    ---
        * N (``int``): Number of detectors.
        * r (``float``): Radius of the detector.
        * r_s (``float``): rₛ Radius of the ``X`` own samples.
        * k (``int``): K number of near neighbors to calculate the average 
        distance of the detectors.
        * metric (``str``): Way to calculate the distance: ``'euclidean', 'minkowski', or 'manhattan'``.
        * max_discards (``int``): This parameter indicates the maximum number of consecutive \
            detector discards, aimed at preventing a possible infinite loop in case a radius is \
            defined that cannot generate non-self detectors.
        * seed (``int``): Seed for the random generation of detector values.
        * algorithm(``str``), Set the algorithm version:

                * ``'default-NSA'``: Default algorithm with fixed radius.
                * ``'V-detector'``: This algorithm uses a variable radius for anomaly detection \
                    in feature spaces.

            Defaults to ``'default-NSA'``.

        * non_self_label (``str``): This variable stores the label that will be when the data has \
            only one output class, and the sample is classified as not belonging to that class. \
            Defaults to ``'non-self'``.
        * cell_bounds (``bool``): If set to ``True``, this option limits the generation of \
            detectors to the space within the plane between 0 and 1. This means that any detector \
            whose radius exceeds this limit is discarded, this variable is only used in the \
            ``V-detector`` algorithm. Defaults to ``False``.
        * p (``float``): This parameter stores the value of ``p`` used in the Minkowski distance. \
            The default is ``2``, which represents normalized Euclidean distance. Different values \
            of p lead to different variants of the Minkowski distance \
            [learn more](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.minkowski.html).

        * detectors (``dict``): This variable stores a list of detectors by class.
        * classes (``npt.NDArray``): list of output classes.

    ---

    A classe ``RNSA`` (Algoritmo de Seleção Negativa de Valor Real) tem a finalidade de classificação \
    e identificação de anomalias através do método self e not self . 

    Attributes:
    ---
        * N (``int``): Quantidade de detectores.
        * r (``float``): Raio do detector.
        * r_s (``float``): O valor de ``rₛ`` é o raio das amostras próprias da matriz ``X``.
        * k (``int``):  K quantidade de vizinhos próximos para calcular a média da distância dos detectores.
        * metric (``str``): Forma de calcular a distância: ``'euclidiana', 'minkowski', or 'manhattan'``.
        * max_discards (``int``): Este parâmetro indica o número máximo de descartes de detectores \
            em sequência, que tem como objetivo evitar um possível loop infinito caso seja definido \
            um raio que não seja possível gerar detectores do não-próprio.
        * seed (``int``): Semente para a geração randômica dos valores dos detectores.
        * algorithm (``str``), Definir a versão do algoritmo:

                * ``'default-NSA'``: Algoritmo padrão com raio fixo.
                * ``'V-detector'``: Este algoritmo utiliza um raio variável para a detecção de \
                    anomalias em espaços de características. 

            Defaults to ``'default-NSA'``.

        * non_self_label (``str``): Esta variável armazena o rótulo que será atribuído quando \
            os dados possuírem apenas uma classe de saída, e a amostra for classificada como não \
            pertencente a essa classe. Defaults to ``'non-self'``.
        * cell_bounds (``bool``):  Se definido como ``True``, esta opção limita a geração dos \
            detectores ao espaço do plano compreendido entre 0 e 1. Isso significa que qualquer \
            detector cujo raio ultrapasse esse limite é descartado, e esta variável é usada \
            exclusivamente no algoritmo ``V-detector``.
        * p (``float``): Este parâmetro armazena o valor de ``p`` utilizada na distância de \
            Minkowski. O padrão é ``2``, o que significa distância euclidiana normalizada. \
            Diferentes valores de p levam a diferentes variantes da distância de Minkowski \
            [saiba mais](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.minkowski.html).

        * detectors (``dict``): Essa variável armazena uma lista com detectores por classes.
        * classes (``npt.NDArray``): lista com as classes de saída.
    """

    def __init__(self, N: int = 100, r: float = 0.05, r_s: float = 0.0001, k: int = 1,
                 metric: Literal["manhattan", "minkowski", "euclidean"] = "euclidean", max_discards: int = 1000,
                 seed: int = None, algorithm: Literal["default-NSA", "V-detector"] = "default-NSA",
                 **kwargs: Dict[str, Union[bool, str, float]]):
        """
        Negative Selection class constructor (``RNSA``).

        Details:
        ---
            This method initializes the ``detectors``, ``classes``, ``k``, ``metric``, ``N``, ``r``, \
            ``r_S``, ``max_discards``, ``seed`` and ``algorithm`` attributes.

        Parameters:
        ---
            * N (``int``): Number of detectors. Defaults to ``100``.
            * r (``float``): Radius of the detector. Defaults to ``0.05``.
            * r_s (``float``): rₛ Radius of the ``X`` own samples. Defaults to ``0.0001``.
            * k (``int``): Number of neighbors near the randomly generated detectors to perform the \
                distance average calculation. Defaults to ``1``.
            * metric (``str``): Way to calculate the distance between the detector and the sample:

                * ``'Euclidean'`` ➜ The calculation of the distance is given by the expression: \
                √( (x₁ – x₂)² + (y₁ – y₂)² + ... + (yn – yn)²).
                * ``'minkowski'`` ➜ The calculation of the distance is given by the expression: \
                ( |X₁ – Y₁|p + |X₂ – Y₂|p + ... + |Xn – Yn|p) ¹/ₚ.
                * ``'manhattan'`` ➜ The calculation of the distance is given by the expression: \
                ( |x₁ – x₂| + |y₁ – y₂| + ... + |yn – yn|) .

            Defaults to ``'euclidean'``.

            * max_discards (``int``): This parameter indicates the maximum number of consecutive \
                detector discards, aimed at preventing a possible infinite loop in case a radius is \
                defined that cannot generate non-self detectors. Defaults to ``1000``.
            * seed (``int``): Seed for the random generation of values in the detectors. Defaults \
                to ``None``.

            * algorithm(``str``), Set the algorithm version:

                * ``'default-NSA'``: Default algorithm with fixed radius.
                * ``'V-detector'``: This algorithm is based on the article \
                    "[Real-Valued Negative Selection Algorithm with Variable-Sized Detectors](https://doi.org/10.1007/978-3-540-24854-5_30)", \
                    by Ji, Z., Dasgupta, D. (2004), and uses a variable radius for anomaly \
                    detection in feature spaces.

            Defaults to ``'default-NSA'``.

            - ``**kwargs``:
                    - non_self_label (``str``): This variable stores the label that will be assigned \
                        when the data has only one output class, and the sample is classified as not \
                        belonging to that class. Defaults to ``'non-self'``.
                    - cell_bounds (``bool``): If set to ``True``, this option limits the generation \
                        of detectors to the space within the plane between 0 and 1. This means that \
                        any detector whose radius exceeds this limit is discarded, this variable is \
                        only used in the ``V-detector`` algorithm. Defaults to ``False``.
                    - p (``float``): This parameter stores the value of ``p`` used in the Minkowski \
                        distance. The default is ``2``, which represents normalized Euclidean distance. \
                        Different values of p lead to different variants of the Minkowski distance \
                        [learn more](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.minkowski.html).
        ---

        Construtor da classe de Seleção negativa (``RNSA``).

        Details:
        ---
            Este método inicializa os atributos ``detectors``, ``classes``, ``k``, ``metric``, ``N``, \
            ``r`` e ``seed``.

        Parameters:
        ---
            * N (``int``): Quantidade de detectores. Defaults to ``100``.
            * r (``float``): Raio do detector. Defaults to ``0.05``.
            * r_s (``float``): O valor de ``rₛ`` é o raio das amostras próprias da matriz ``X``. \
                Defaults to ``0.0001``.
            * k (``int``): Quantidade de vizinhos próximos dos detectores gerados aleatoriamente \
                para efetuar o cálculo da média da distância. Defaults to ``1``.
            * metric (``str``): Forma para se calcular a distância entre o detector e a amostra: 

                * ``'euclidiana'`` ➜ O cálculo da distância dá-se pela expressão: \
                    √( (x₁ – x₂)² + (y₁ – y₂)² + ... + (yn – yn)²).
                * ``'minkowski'``  ➜ O cálculo da distância dá-se pela expressão: \
                    ( |X₁ – Y₁|p + |X₂ – Y₂|p + ... + |Xn – Yn|p) ¹/ₚ.
                * ``'manhattan'``  ➜ O cálculo da distância dá-se pela expressão: \
                    ( |x₁ – x₂| + |y₁ – y₂| + ... + |yn – yn|).

            Defaults to ``'euclidean'``.

            * max_discards (``int``): Este parâmetro indica o número máximo de descartes de detectores \
                em sequência, que tem como objetivo evitar um possível loop infinito caso seja definido \
                um raio que não seja possível gerar detectores do não-próprio. Defaults to ``1000``.
            * seed (``int``): Semente para a geração randômica dos valores nos detectores. \
                Defaults to ``None``.
            * algorithm (``str``), Definir a versão do algoritmo:

                * ``'default-NSA'``: Algoritmo padrão com raio fixo.
                * ``'V-detector'``: Este algoritmo é baseado no artigo \
                    "[Real-Valued Negative Selection Algorithm with Variable-Sized Detectors](https://doi.org/10.1007/978-3-540-24854-5_30)", \
                    de autoria de Ji, Z., Dasgupta, D. (2004), e utiliza um raio variável para a \
                    detecção de anomalias em espaços de características. 

            Defaults to ``'default-NSA'``.

            - ``**kwargs``:
                    - non_self_label (``str``): Esta variável armazena o rótulo que será atribuído \
                        quando os dados possuírem apenas uma classe de saída, e a amostra for \
                        classificada como não pertencente a essa classe. Defaults to ``'non-self'``.
                    - cell_bounds (``bool``):  Se definido como ``True``, esta opção limita a \
                        geração dos detectores ao espaço do plano compreendido entre 0 e 1. Isso \
                        significa que qualquer detector cujo raio ultrapasse esse limite é descartado, \
                        e esta variável é usada exclusivamente no algoritmo ``V-detector``.
                    - p (``float``): Este parâmetro armazena o valor de ``p`` utilizada na distância \
                        de Minkowski. O padrão é ``2``, o que significa distância euclidiana normalizada. \
                        Diferentes valores de p levam a diferentes variantes da distância de Minkowski \
                        [saiba mais](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.minkowski.html).
        """

        super().__init__(metric)
        if metric == "manhattan" or metric == "minkowski":
            self.metric = metric
        else:
            self.metric = "euclidean"

        if seed is not None and isinstance(seed, int):
            np.random.seed(seed)
            self.seed: int = seed
        else:
            self.seed = None

        if k < 1:
            self.k: int = 1
        else:
            self.k: int = k

        if N < 1:
            self.N: int = 100
        else:
            self.N: int = N

        if r < 0:
            self.r: float = 0.05
        else:
            self.r: float = r

        if r_s > 0:
            self.r_s: float = r_s
        else:
            self.r_s: float = 0

        if algorithm == "V-detector":
            self._Detector = namedtuple("Detector", "position radius")
            self._algorithm: str = algorithm
        else:
            self._Detector = namedtuple("Detector", "position")
            self._algorithm: str = "default-NSA"

        if max_discards > 0:
            self.max_discards: int = max_discards
        else:
            self.max_discards: int = 1000

        # Obtém as variáveis do kwargs.
        self.p: float = kwargs.get("p", 2)
        self._cell_bounds: bool = kwargs.get("cell_bounds", False)
        self.non_self_label: str = kwargs.get("non_self_label", "non-self")

        # Inicializa as demais variáveis da classe como None
        self.detectors: Union[dict, None] = None
        self.classes: npt.NDArray = None

    def fit(self, X: npt.NDArray, y: npt.NDArray, verbose: bool = True):
        """
        The function ``fit(...)``, performs the training according to ``X`` and ``y``, using the method
        negative selection method(``NegativeSelect``).

        Parameters:
        ---
            * X (``npt.NDArray``): Training array, containing the samples and their characteristics, \
                [``N samples`` (rows)][``N features`` (columns)].
            * y (``npt.NDArray``): Array of target classes of ``X`` with [``N samples`` (lines)].
            * verbose (``bool``): Feedback from detector generation to the user.
        returns:
        ---
            (``self``): Returns the instance itself.

        ----

        A função ``fit(...)``, realiza o treinamento de acordo com ``X`` e ``y``, usando o método
        de seleção negativa(``NegativeSelect``).

        Parameters:
        ---
            * X (``npt.NDArray``): Array de treinamento, contendo as amostras é suas características, \
                [``N amostras`` (linhas)][``N características`` (colunas)].
            * y (``npt.NDArray``):  Array com as classes alvos de ``X`` com [``N amostras`` (linhas)].
            * verbose (``bool``): Feedback da geração de detectores para o usuário.
        Returns:
        ---
            (``self``): Retorna a própria instância.
        """
        super()._check_and_raise_exceptions_fit(X, y)

        # Identificando as classes possíveis, dentro do array de saídas ``y``.
        self.classes = np.unique(y)
        # Dict que armazenará os detectores com as classes como key.
        list_detectors_by_class = dict()
        # Separa as classes para o treinamento.
        sample_index = self.__slice_index_list_by_class(y)
        # Barra de progresso para a geração de todos os detectores.
        if verbose:
            progress = tqdm(total=int(self.N * (len(self.classes))),
                            bar_format="{desc} ┇{bar}┇ {n}/{total} detectors", postfix="\n",)
        for _class_ in self.classes:
            # Inicia o conjunto vazio que conterá os detectores válidos.
            valid_detectors_set = []
            discard_count = 0
            # Informando em qual classe o algoritmo está para a barra de progresso.
            if verbose:
                progress.set_description_str(
                    f"Generating the detectors for the {_class_} class:"
                )
            while len(valid_detectors_set) < self.N:
                # Gera um vetor candidato a detector aleatoriamente com valores entre 0 e 1.
                vector_x = np.random.random_sample(size=X.shape[1])
                # Verifica a validade do detector para o não-próprio com relação às amostras da classe.
                valid_detector = self.__checks_valid_detector(
                    X=X, vector_x=vector_x, samples_index_class=sample_index[_class_]
                )

                # Se o detector for válido, adicione a lista dos válidos.
                if self._algorithm == "V-detector" and valid_detector is not False:
                    discard_count = 0
                    valid_detectors_set.append(
                        self._Detector(vector_x, valid_detector[1])
                    )
                    if verbose:
                        progress.update(1)
                elif valid_detector:
                    discard_count = 0
                    valid_detectors_set.append(self._Detector(vector_x))
                    if verbose:
                        progress.update(1)
                else:
                    discard_count += 1
                    if discard_count == self.max_discards:
                        raise Exception(
                            "An error has been identified:\n"
                            f"the maximum number of discards of detectors for the {_class_} class "
                            "has been reached.\nIt is recommended to check the defined radius and "
                            "consider reducing its value."
                        )

            # Adicionar detectores, com as classes como chave na dict.
            list_detectors_by_class[_class_] = valid_detectors_set
        # Informar a finalização da geração dos detectores para as classes.
        if verbose:
            progress.set_description(
                f'\033[92m✔ Non-self detectors for classes ({", ".join(map(str, self.classes))}) '
                f'successfully generated\033[0m'
            )
        # Armazena os detectores encontrados no atributo, para os detectores da classe.
        self.detectors = list_detectors_by_class
        return self

    def predict(self, X: npt.NDArray) -> Optional[npt.NDArray]:
        """
        Function to perform the prediction of classes based on detectors
        created after training.

        Parameters:
        ---
            * X (``npt.NDArray``): Array with input samples with [``N samples`` (Lines)] and
            [``N characteristics``(Columns)]

        returns:
        ---
            * C – (``npt.NDArray``): an ndarray of the form ``C`` [``N samples``],
            containing the predicted classes for ``X``.
            * ``None``: If there are no detectors for the prediction.

        ---

        Função para efetuar a previsão das classes com base nos detectores
        criados após o treinamento.

        Parameters:
        ---
            * X (``npt.NDArray``): Array com as amostras de entradas com [``N amostras`` (Linhas)] e
            [``N características``(Colunas)]

        Returns:
        ---
            * C – (``npt.NDArray``): um ndarray de forma ``C`` [``N amostras``],
            contendo as classes previstas para ``X``.
            * ``None``: Se não existir detectores para a previsão.
        """
        # se não houver detectores retorna None.
        if self.detectors is None:
            return None
        elif not isinstance(X, (np.ndarray, list)):
            raise TypeError("X is not an ndarray or list")
        elif len(self.detectors[self.classes[0]][0].position) != len(X[0]):
            raise Exception(
                "X does not have {} features to make the prediction".format(
                    len(self.detectors[self.classes[0]][0])
                )
            )

        # Inicia um array vazio.
        C = np.empty(shape=0)
        # Para cada linha de amostra em X.
        for line in X:
            class_found: bool
            _class_ = self.__compare_sample_to_detectors(line)
            if _class_ is None:
                class_found = False
            else:
                C = np.append(C, [_class_])
                class_found = True

            # Se possuir apenas uma classe e não classificar a amostra define a saída como não-própria.
            if not class_found and len(self.classes) == 1:
                C = np.append(C, [self.non_self_label])
            # Se não identificar a classe com os detectores, coloca a classe com a maior distância
            # da média dos seus detectores.
            elif not class_found:
                average_distance: dict = {}
                for _class_ in self.classes:
                    detectores = list(
                        map(lambda x: x.position, self.detectors[_class_])
                    )
                    average_distance[_class_] = np.average(
                        [self.__distance(detector, line)
                         for detector in detectores]
                    )
                C = np.append(C, [max(average_distance, key=average_distance.get)])
        return C

    def __checks_valid_detector(self, X: npt.NDArray = None, vector_x: npt.NDArray = None,
                                    samples_index_class: npt.NDArray = None):
        """
        Function to check if the detector has a valid non-proper ``r`` radius for the class.

        Parameters:
        ---
        * X (``npt.NDArray``): Array ``X`` with the samples.
        * vector_x (``npt.NDArray``): Randomly generated vector x candidate detector with values \
              ​​between [0, 1].
        * samples_index_class (``npt.NDArray``): Sample positions of a class in ``X``.

        returns:
        ---
            * Validity (``bool``): Returns whether the detector is valid or not.

        ---

        Função para verificar se o detector possui raio ``r`` válido do não-próprio para a classe.

        Parameters:
        ---
        * X (``npt.NDArray``): Array ``X`` com as amostras.
        * vector_x (``npt.NDArray``): Vetor x candidato a detector gerado aleatoriamente com valores \
            entre [0, 1].
        * samples_index_class (``npt.NDArray``): Posições das amostras de uma classe em ``X``.

        Returns:
        ---
            * Validade (``bool``): Retorna se o detector é válido ou não.

        """
        # Se um ou mais array de entrada possuir zero dados, retorna falso.
        if (np.size(samples_index_class) == 0 or np.size(X) == 0 or np.size(vector_x) == 0):
            return False
        # se self.k > 1  utiliza os k vizinhos mais próximos (knn), se não verifica o detector sem
        # considerar os knn.
        if self.k > 1:
            # Iniciar a lista dos knn vazia.
            knn_list = np.empty(shape=0)
            for i in samples_index_class:
                # Calcula a distância entre os dois vetores e adiciona a lista dos knn, se a
                # distância for menor que a maior da lista.
                knn_list = self.__compare_KnearestNeighbors_List(
                    knn_list, self.__distance(X[i], vector_x)
                )
            # Se a média das distâncias na lista dos knn, for menor que o raio, retorna verdadeiro.
            distance_mean = np.mean(knn_list)
            if self._algorithm == "V-detector":
                return self.__detector_is_valid_to_Vdetector(distance_mean, vector_x)
            elif distance_mean > (self.r + self.r_s):
                return True  # Detector é valido!
        else:
            distance: Union[float, None] = None
            for i in samples_index_class:
                if self._algorithm == "V-detector":
                    new_distance = self.__distance(X[i], vector_x)
                    if distance is None:
                        distance = new_distance
                    elif distance > new_distance:
                        distance = new_distance
                else:
                    # Calcula a distância entre os vetores, se menor ou igual ao raio + raio da
                    # amostra define a validade do detector como falso.
                    if (self.r + self.r_s) >= self.__distance(X[i], vector_x):
                        return False  # Detector não é valido!

            if self._algorithm == "V-detector":
                return self.__detector_is_valid_to_Vdetector(distance, vector_x)
            return True  # Detector é valido!

        return False  # Detector não é valido!

    def __compare_KnearestNeighbors_List(self, knn: npt.NDArray, distance: float) -> npt.NDArray:
        """
        Compares the k-nearest neighbor distance at position ``k-1`` in the list ``knn``,
        if the distance of the new sample is less, replace it and sort in ascending order.


        Parameters:
        ---
            knn (npt.NDArray): List of k-nearest neighbor distances.
            distance (float): Distance to check.

        returns:
        ---
            npt.NDArray: Updated and sorted nearest neighbor list.

        ---

        Compara a distância do k-vizinho mais próximo na posição ``k-1``da lista ``knn``,
        se a distância da nova amostra for menor, substitui ela e ordena em ordem crescente.


        Parameters:
        ---
            knn (npt.NDArray): Lista de distâncias dos k-vizinhos mais próximos.
            distance (float): Distância a ser verificada.

        Returns:
        ---
            npt.NDArray: Lista de vizinhos mais próximos atualizada e ordenada.
        """
        # Se a quantidade de distâncias em knn, for menor que k, adiciona a distância.
        if len(knn) < self.k:
            knn = np.append(knn, distance)
            knn.sort()
        else:
            # Se não, adicione a distância, se a nova distancia for menor que a maior distância da lista.
            if knn[self.k - 1] > distance:
                knn[self.k - 1] = distance
                knn.sort()

        return knn

    def __compare_sample_to_detectors(self, line: npt.NDArray):
        """
        Function to compare a sample with the detectors, verifying if the sample is proper.

        Details:
        ---
        In this function, when there is class ambiguity, it returns the class that has the greatest
        average distance between the detectors.

        Parameters:
        ---
            * line: vector with N-features

        returns:
        ---
            * Returns the predicted class with the detectors or None if the sample does not qualify \
                for any class.

        ---

        Função para comparar uma amostra com os detectores, verificando se a amostra é própria.

        Details:
        ---
        Nesta função, quando possui ambiguidade de classes, retorna a classe que possuir a média de \
        distância maior entre os detectores.

        Parameters:
        ---
            * line: vetor com N-características

        Returns:
        ---
            * Retorna a classe prevista com os detectores ou None se a amostra não se qualificar \
                a nenhuma classe.
        """

        # Lista para armazenar as classes e a distância média entre os detectores e a amostra.
        possible_classes = []
        for _class_ in self.classes:
            # Variável para identificar, se a classe foi encontrada com os detectores.
            class_found: bool = True
            sum_distance = 0  # Variável para fazer o somatório das distâncias.
            for detector in self.detectors[_class_]:
                # Calcula a distância entre a amostra e os detectores.
                distance = self.__distance(detector.position, line)
                # Soma as distâncias para calcular a média.
                sum_distance += distance
                if self._algorithm == "V-detector":
                    if distance <= detector.radius:
                        class_found = False
                        break
                elif distance <= self.r:
                    class_found = False
                    break

            # Se a amostra passar por todos os detectores de uma classe, adiciona a classe como
            # possível previsão.
            if class_found:
                possible_classes.append([_class_, sum_distance / self.N])
        # Se classificar como pertencentes a apenas uma classe, retorna a classe.
        if len(possible_classes) == 1:
            return possible_classes[0][0]
        # Se, pertencer a mais de uma classe, retorna a classe com a distância média mais distante.
        elif len(possible_classes) > 1:
            return max(possible_classes, key=lambda x: x[1])[0]
        else:  # Se não, retorna None
            return None

    def __distance(self, u: npt.NDArray, v: npt.NDArray):
        """
        Function to calculate the distance between two points by the chosen ``metric``.

        Parameters:
        ---
            * u (``npt.NDArray``): Coordinates of the first point.
            * v (``npt.NDArray``): Coordinates of the second point.

        returns:
        ---
            * Distance (``double``) between the two points.

        ---

        Função para calcular a distância entre dois pontos pela ``metric`` escolhida.

        Parameters:
        ---
            * u (``npt.NDArray``): Coordenadas do primeiro ponto.
            * v (``npt.NDArray``): Coordenadas do segundo ponto.

        Returns:
        ---
            * Distância (``double``) entre os dois pontos.
        """
        return super()._distance(u, v)

    def __detector_is_valid_to_Vdetector(self, distance: float, vector_x: npt.NDArray):
        """
        Check if the distance between the detector and the samples, minus the radius of the samples,
        is greater than the minimum radius.

        Parameters:
        ---
        distance (``float``): minimum distance calculated between all samples.
        vector_x (``numpy.ndarray``): randomly generated candidate detector vector x with values \
        between 0 and 1.

        Returns:
        ---
        * ``False``: if the calculated radius is smaller than the minimum distance or exceeds the \
            edge of the space, if this option is enabled.
        * ``True`` and the distance minus the radius of the samples, if the radius is valid.`

        ----

        Verifique se a distância entre o detector e as amostras, descontando o raio das amostras, \
        é maior do que o raio mínimo.

        Parameters:
        ---
        distance (``float``): distância mínima calculada entre todas as amostras.
        vector_x (``numpy.ndarray``): vetor x candidato do detector gerado aleatoriamente, com \
        valores entre 0 e 1.

        Returns:
        ---

        * ``False``: caso o raio calculado seja menor do que a distância mínima ou ultrapasse a \
            borda do espaço, caso essa opção esteja habilitada.
        * ``True`` e a distância menos o raio das amostras, caso o raio seja válido.
        """
        new_detector_r = float(distance - self.r_s)
        if self.r >= new_detector_r:
            return False  # Detector não é valido!
        else:
            # se _cell_bounds igual a True, considera o detector esta dentro do limite do plano.
            if self._cell_bounds:
                for p in vector_x:
                    if (p - new_detector_r) < 0 or (p + new_detector_r) > 1:
                        return False
            return True, new_detector_r

    def __slice_index_list_by_class(self, y: npt.NDArray) -> dict:
        """
        The function ``__slice_index_list_by_class(...)``, separates the indices of the lines \
        according to the output class, to loop through the sample array, only in positions where \
        the output is the class being trained.

        Parameters:
        ---
            * y (npt.NDArray): Receives a ``y``[``N sample``] array with the output classes of the \
                ``X`` sample array.

        returns:
        ---
            * dict: A dictionary with the list of array positions(``y``), with the classes as key.

        ---

        A função ``__slice_index_list_by_class(...)``, separa os índices das linhas conforme a classe \
        de saída, para percorrer o array de amostra, apenas nas posições que a saída for a classe que \
        está sendo treinada.

        Parameters:
        ---
            * y (npt.NDArray): Recebe um array ``y``[``N amostra``] com as classes de saída do array \
                de amostra ``X``.

        Returns:
        ---
            * dict: Um dicionário com a lista de posições do array(``y``), com as classes como chave.
        """
        return super()._slice_index_list_by_class(y)

    def score(self, X: npt.NDArray, y: list) -> float:
        """
        Score function calculates forecast accuracy.

        Details:
        ---
        This function performs the prediction of X and checks how many elements are equal between \
        vector y and y_predicted. This function was added for compatibility with some scikit-learn \
        functions.

        Parameters:
        -----------

        * X (np.ndarray): Feature set with shape (n_samples, n_features).
        * y (np.ndarray): True values with shape (n_samples,).

        Returns:
        -------

        accuracy: float
            The accuracy of the model.

        ---

        Função score calcular a acurácia da previsão.

        Details:
        ---
        Esta função realiza a previsão de X e verifica quantos elementos são iguais entre o vetor \
        y e y_previsto. Essa função foi adicionada para oferecer compatibilidade com algumas \
        funções do scikit-learn.

        Parameters:
        ---

        * X (np.ndarray): Conjunto de características com shape (n_samples, n_features).
        * y (np.ndarray): Valores verdadeiros com shape (n_samples,).

        returns:
        ---

        * accuracy (float): A acurácia do modelo.

        """
        return super()._score(X, y)

    def get_params(self, deep: bool = True) -> dict:
        return {
            "N": self.N,
            "r": self.r,
            "k": self.k,
            "metric": self.metric,
            "seed": self.seed,
            "algorithm": self._algorithm,
            "r_s": self.r_s,
            "cell_bounds": self._cell_bounds,
            "p": self.p,
        }


class BNSA(Base):
    """
    The ``BNSA`` (Binary Negative Selection Algorithm) class is for classification and identification \
    purposes of anomalies through the self and not self method.

    Attributes:
    ---

        * N (``int``): Number of detectors.
        * aff_thresh (``float``): The variable represents the percentage of similarity between the \
            T cell and the own samples.
        * max_discards (``int``): This parameter indicates the maximum number of detector discards \
            in sequence, which aims to avoid a possible infinite loop if a radius is defined that \
            it is not possible to generate non-self detectors.
        * seed (``int``): Seed for the random generation of values in the detectors.

        * detectors (``dict``): This variable stores a list of detectors by class.
        * classes (``npt.NDArray``): list of output classes.


    ---

    A classe ``BNSA`` (Algoritmo de Seleção Negativa Binária) tem a finalidade de classificação e \
    identificação de anomalias através do método self e not self .

    Attributes:
    ---
        * N (``int``): Quantidade de detectores. Defaults to ``100``.
        * aff_thresh (``float``): A variável representa a porcentagem de similaridade entre a célula \
            T e as amostras próprias. O valor padrão é de 10% (0,1), enquanto que o valor de 1,0 \
            representa 100% de similaridade.
        * max_discards (``int``): Este parâmetro indica o número máximo de descartes de detectores \
            em sequência, que tem como objetivo evitar um possível loop infinito caso seja definido \
            um raio que não seja possível gerar detectores do não-próprio. Defaults to ``100``.
        * seed (``int``): Semente para a geração randômica dos valores nos detectores. Defaults to ``None``.
        * no_label_sample_selection (``str``): Method for selecting labels for samples designated as \
            non-members by all non-member detectors. Defaults to ``max_average_difference``.


        * detectors (``dict``): Essa variável armazena uma lista com detectores por classes.
        * classes (``npt.NDArray``): lista com as classes de saída.

    """

    def __init__(self, N: int = 100, aff_thresh: float = 0.1, max_discards: int = 1000, seed: int = None,
                 no_label_sample_selection: Literal["max_average_difference", "max_nearest_difference"] =
                 "max_average_difference"):
        """
        Constructor of the Negative Selection class (``BNSA``).

         Details:
         ---
             This method initializes the ``detectors``, ``classes``, ``N``, ``t`` and ``seed`` attributes.

         Parameters:
         ---
             * N (``int``): Number of detectors. Defaults to ``100``.
             * aff_thresh (``float``): The variable represents the percentage of similarity between \
                the T cell and the own samples. The default value is 10% (0.1), while a value of 1.0 \
                represents 100% similarity.
             * max_discards (``int``): This parameter indicates the maximum number of detector \
                discards in sequence, which aims to avoid a possible infinite loop if a radius is \
                defined that it is not possible to generate non-self detectors. Defaults to ``1000``.
             * seed (``int``): Seed for the random generation of values in the detectors. Defaults to ``None``.
             * no_label_sample_selection (``str``): Method for selecting labels for samples designated as \
                non-members by all non-member detectors. Available method types:
                    - (``max_average_difference``): Selects the class with the highest average difference \
                        among the detectors.
                    - (``max_nearest_difference``): Selects the class with the highest difference between \
                        the nearest and farthest detector from the sample.
        ---

        Construtor da classe de Seleção negativa (``BNSA``).

        Details:
        ---
            Este método inicializa os atributos ``detectors``, ``classes``, ``N``, ``t`` e ``seed``.

        Parameters:
        ---
            * N (``int``): Quantidade de detectores. Defaults to ``100``.
            * aff_thresh (``float``): A variável representa a porcentagem de similaridade entre a \
                célula T e as amostras próprias. O valor padrão é de 10% (0,1), enquanto que o valor \
                de 1,0 representa 100% de similaridade.
            * max_discards (``int``): Este parâmetro indica o número máximo de descartes de detectores \
                em sequência, que tem como objetivo evitar um possível loop infinito caso seja definido \
                um raio que não seja possível gerar detectores do não-próprio. Defaults to ``1000``.
            * seed (``int``): Semente para a geração randômica dos valores nos detectores. Defaults to ``None``.
            * no_label_sample_selection (``str``): Método para a seleção de rótulos para amostras designadas \
                como não pertencentes por todos os detectores não pertencentes. Tipos de métodos disponíveis:
                    - (``max_average_difference``): Seleciona a classe com a maior diferença média entre os \
                        detectores.
                    - (``max_nearest_difference``): Seleciona a classe com a maior diferença entre o detector \
                        mais próximo e mais distante da amostra.

        """
        super().__init__()
        if N > 0:
            self.N: int = N
        else:
            self.N: int = 100

        if 0 < aff_thresh < 1:
            self.aff_thresh: float = aff_thresh
        else:
            self.aff_thresh: float = 0.1
        if max_discards > 0:
            self.max_discards: int = max_discards
        else:
            self.max_discards: int = 1000

        if seed is not None and isinstance(seed, int):
            np.random.seed(seed)
            self.seed: int = seed
        else:
            self.seed = None

        if no_label_sample_selection == 'nearest_difference':
            self.no_label_sample_selection = 'nearest_difference'
        else:
            self.no_label_sample_selection = 'max_average_difference'

        self.classes: npt.NDArray = None
        self.detectors: npt.NDArray = None

    def fit(self, X: npt.NDArray, y: npt.NDArray, verbose: bool = True):
        """
        The function ``fit(...)``, performs the training according to ``X`` and ``y``, using the method
        negative selection method(``NegativeSelect``).

        Parameters:
        ---
            * X (``npt.NDArray``): Training array, containing the samples and their characteristics,
            [``N samples`` (rows)][``N features`` (columns)].
            * y (``npt.NDArray``): Array of target classes of ``X`` with [``N samples`` (lines)].
            * verbose (``bool``): Feedback from detector generation to the user.
        returns:
        ---
            (``self``): Returns the instance itself.

        ----

        A função ``fit(...)``, realiza o treinamento de acordo com ``X`` e ``y``, usando o método
        de seleção negativa(``NegativeSelect``).

        Parameters:
        ---
            * X (``npt.NDArray``): Array de treinamento, contendo as amostras é suas características, \
                [``N amostras`` (linhas)][``N características`` (colunas)].
            * y (``npt.NDArray``):  Array com as classes alvos de ``X`` com [``N amostras`` (linhas)].
            * verbose (``bool``): Feedback da geração de detectores para o usuário.
        Returns:
        ---
            (``self``): Retorna a própria instância.
        """
        super()._check_and_raise_exceptions_fit(X, y, "BNSA")

        # Converte todo o array X para boolean
        if X.dtype != bool:
            X = X.astype(bool)

        # Identificando as classes possíveis, dentro do array de saídas ``y``.
        self.classes = np.unique(y)
        # Dict que armazenará os detectores com as classes como key.
        list_detectors_by_class = dict()
        # Separa as classes para o treinamento.
        sample_index: dict = self.__slice_index_list_by_class(y)
        # Barra de progresso para a geração de todos os detectores.
        if verbose:
            progress = tqdm(total=int(self.N * (len(self.classes))),
                            bar_format='{desc} ┇{bar}┇ {n}/{total} detectors', postfix='\n')

        for _class_ in self.classes:
            # Inicia o conjunto vazio que conterá os detectores válidos.
            valid_detectors_set: list = []
            discard_count: int = 0
            # Informando em qual classe o algoritmo está para a barra de progresso.
            if verbose:
                progress.set_description_str(
                    f"Generating the detectors for the {_class_} class:")
            while len(valid_detectors_set) < self.N:

                is_valid_detector: bool = True
                # Gera um vetor candidato a detector aleatoriamente com valores 0 e 1.
                vector_x = np.random.choice([False, True], size=X.shape[1])
                # Calcula a distância entre o candidato e as amostras da classe.
                distances = cdist(np.expand_dims(vector_x, axis=0), 
                                  X[sample_index[_class_]], metric='hamming')
                # Verifica se alguma das distâncias está abaixo ou igual ao limiar
                is_valid_detector = not np.any(distances <= self.aff_thresh)

                # Se o detector for válido, adicione a lista dos válidos.
                if is_valid_detector:
                    discard_count = 0
                    valid_detectors_set.append(vector_x)
                    if verbose:
                        progress.update(1)
                else:
                    discard_count += 1
                    if discard_count == self.max_discards:
                        raise Exception(
                            "An error has been identified:\n"
                            f"the maximum number of discards of detectors for the {_class_} "
                            "class has been reached.\nIt is recommended to check the defined "
                            "radius and consider reducing its value."
                        )

            # Adicionar detectores, com as classes como chave na dict.
            list_detectors_by_class[_class_] = valid_detectors_set

        # Informar a finalização da geração dos detectores para as classes.
        if verbose:
            progress.set_description(
                f'\033[92m✔ Non-self detectors for classes ({", ".join(map(str, self.classes))}) '
                f'successfully generated\033[0m')
        # Armazena os detectores encontrados no atributo, para os detectores da classe.
        self.detectors = list_detectors_by_class
        return self

    def predict(self, X: npt.NDArray) -> Optional[npt.NDArray]:
        """
        Function to perform the prediction of classes based on detectors
        created after training.

        Parameters:
        ---
            * X (``npt.NDArray``): Array with input samples with [``N samples`` (Lines)] and
            [``N characteristics``(Columns)]

        returns:
        ---
            * C – (``npt.NDArray``): an ndarray of the form ``C`` [``N samples``],
            containing the predicted classes for ``X``.
            * ``None``: If there are no detectors for the prediction.

        ---

        Função para efetuar a previsão das classes com base nos detectores
        criados após o treinamento.

        Parameters:
        ---
            * X (``npt.NDArray``): Array com as amostras de entradas com [``N amostras`` (Linhas)] e
            [``N características``(Colunas)]

        Returns:
        ---
            * C – (``npt.NDArray``): um ndarray de forma ``C`` [``N amostras``],
            contendo as classes previstas para ``X``.
            * ``None``: Se não existir detectores para a previsão.
        """
        # se não houver detectores retorna None.
        if self.detectors is None:
            return None
        elif not isinstance(X, (np.ndarray, list)):
            raise TypeError("X is not an ndarray or list")
        elif len(self.detectors[self.classes[0]][0]) != len(X[0]):
            raise Exception(
                "X does not have {} features to make the prediction".format(
                    len(self.detectors[self.classes[0]][0])
                )
            )
        # Verifica se a matriz X contém apenas amostras binárias. Caso contrário, lança uma exceção.
        if not np.isin(X, [0, 1]).all():
            raise ValueError(
                "The array X contains values that are not composed only of 0 and 1."
            )

        # Converte todo o array X para boolean
        if X.dtype != bool:
            X = X.astype(bool)

        # Inicia um array vazio.
        C = np.empty(shape=0)
        # Para cada linha de amostra em X.
        for line in X:
            class_found: bool = True
            # Lista para armazenar as possíveis classes às quais a amostra se adequou ao self na
            # comparação com os detectores non-self.
            possible_classes: list = []
            for _class_ in self.classes:
                # Lista para armazenar as taxas de similaridade entre a amostra e os detectores.
                similarity_sum: float = 0

                # Calcula a distância de Hamming entre a linha e todos os detectores
                distances = cdist(np.expand_dims(line, axis=0),
                                  self.detectors[_class_], metric='hamming')

                # Verificar se alguma distância está abaixo ou igual ao limiar
                if np.any(distances <= self.aff_thresh):
                    class_found = False
                else:
                    # Somar todas as distâncias
                    similarity_sum = np.sum(distances)

                # Se a amostra passar por todos os detectores de uma classe, adiciona a classe como
                # possível previsão e sua media de similaridade.
                if class_found:
                    possible_classes.append([_class_, similarity_sum / self.N])

            # Se, pertencer a uma ou mais classes, adiciona a classe com a distância média mais distante.
            if len(possible_classes) > 0:
                C = np.append(
                    C, [max(possible_classes, key=lambda x: x[1])[0]])
                class_found = True
            else:
                class_found = False

            # Se possuir apenas uma classe e não classificar a amostra define a saída como não-própria.
            if not class_found and len(self.classes) == 1:
                C = np.append(C, ["non-self"])
            # Se a classe não puder ser identificada pelos detectores
            elif not class_found:
                class_differences: dict = {}
                for _class_ in self.classes:
                    # Atribua-a o rotulo a classe com à maior distância em relação ao detector mais próximo.
                    if self.no_label_sample_selection == 'nearest_difference':
                        difference_min: float = cdist( np.expand_dims(line, axis=0),
                                                    self.detectors[_class_], metric='hamming'
                                                ).min()
                        class_differences[_class_] = difference_min
                    # Ou com base na maior distância com relação à média da distancias dos detectores
                    else:
                        difference_sum: float = cdist( np.expand_dims(line, axis=0),
                                                    self.detectors[_class_], metric='hamming'
                                                ).sum()
                        class_differences[_class_] = difference_sum / self.N

                C = np.append(C, [max(class_differences, key=class_differences.get)])

        return C

    def __slice_index_list_by_class(self, y: npt.NDArray) -> dict:
        """
        The function ``__slice_index_list_by_class(...)``, separates the indices of the lines according \
        to the output class, to loop through the sample array, only in positions where the output is \
        the class being trained.

        Parameters:
        ---
            * y (npt.NDArray): Receives a ``y``[``N sample``] array with the output classes of the \
                ``X`` sample array.

        returns:
        ---
            * dict: A dictionary with the list of array positions(``y``), with the classes as key.

        ---

        A função ``__slice_index_list_by_class(...)``, separa os índices das linhas conforme a classe \
        de saída, para percorrer o array de amostra, apenas nas posições que a saída for a classe que \
        está sendo treinada.

        Parameters:
        ---
            * y (npt.NDArray): Recebe um array ``y``[``N amostra``] com as classes de saída do array \
                de amostra ``X``.

        Returns:
        ---
            * dict: Um dicionário com a lista de posições do array(``y``), com as classes como chave.
        """
        return super()._slice_index_list_by_class(y)

    def score(self, X: npt.NDArray, y: list) -> float:
        """
        Score function calculates forecast accuracy.

        Details:
        ---
        This function performs the prediction of X and checks how many elements are equal between vector \
        y and y_predicted. This function was added for compatibility with some scikit-learn functions.

        Parameters:
        -----------

        X: np.ndarray
            Feature set with shape (n_samples, n_features).
        y: np.ndarray
            True values with shape (n_samples,).

        Returns:
        -------

        accuracy: float
            The accuracy of the model.

        ---

        Função score calcular a acurácia da previsão.

        Details:
        ---
        Esta função realiza a previsão de X e verifica quantos elementos são iguais entre o vetor y \
        e y_previsto.
        Essa função foi adicionada para oferecer compatibilidade com algumas funções do scikit-learn.

        Parameters:
        ---

        * X : np.ndarray
            Conjunto de características com shape (n_samples, n_features).
        * y : np.ndarray
            Valores verdadeiros com shape (n_samples,).

        returns:
        ---

        accuracy : float
            A acurácia do modelo.
        """
        return super()._score(X, y)

    def get_params(self, deep: bool = True) -> dict:
        return {
            "N": self.N,
            "aff_thresh": self.aff_thresh,
            "max_discards": self.max_discards,
            "seed": self.seed,
        }
