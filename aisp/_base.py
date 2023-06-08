import numpy as np
import numpy.typing as npt
from typing import Literal
from scipy.spatial.distance import euclidean, cityblock, minkowski

class Base:
    """
    The base class contains functions that are used by more than one class in the package, 
    and therefore are considered essential for the overall functioning of the system.

    ---

    A classe base contém funções que são utilizadas por mais de uma classe do pacote, 
    e por isso são consideradas essenciais para o funcionamento geral do sistema.
    """
    def __init__(self, metric: str = 'euclidean'):
        """
        Parameters:
        ---
        * metric (``str``): Way to calculate the distance between the detector and the sample:

                * ``'Euclidean'`` ➜ The calculation of the distance is given by the expression: √( (x₁ – x₂)² + (y₁ – y₂)² + ... + (yn – yn)²).
                * ``'minkowski'`` ➜ The calculation of the distance is given by the expression: ( |X₁ – Y₁|p + |X₂ – Y₂|p + ... + |Xn – Yn|p) ¹/ₚ , In this project ``p == 2``.
                * ``'manhattan'`` ➜ The calculation of the distance is given by the expression: ( |x₁ – x₂| + |y₁ – y₂| + ... + |yn – yn|) .
        
        ---

        Parameters:
        ---
        * metric (``str``): Forma para se calcular a distância entre o detector e a amostra: 

                * ``'euclidiana'`` ➜ O cálculo da distância dá-se pela expressão: √( (x₁ – x₂)² + (y₁ – y₂)² + ... + (yn – yn)²).
                * ``'minkowski'``  ➜ O cálculo da distância dá-se pela expressão: ( |X₁ – Y₁|p + |X₂ – Y₂|p + ... + |Xn – Yn|p) ¹/ₚ , Neste projeto ``p == 2``.
                * ``'manhattan'``  ➜ O cálculo da distância dá-se pela expressão: ( |x₁ – x₂| + |y₁ – y₂| + ... + |yn – yn|).

            Defaults to ``'euclidean'``.
        """
        if metric == 'manhattan' or metric == 'minkowski' or metric == 'euclidean':
            self.metric = metric
        else:
            self.metric = 'euclidean'

    def _distance(self, u: npt.NDArray, v: npt.NDArray):
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
        if self.metric == 'manhattan':
            return cityblock(u, v)
        elif self.metric == 'minkowski':
            return minkowski(u, v, 2)
        else:
            return euclidean(u, v)
        
    def _check_and_raise_exceptions_fit(self, X: npt.NDArray = None, y: npt.NDArray = None, _class_: Literal['RNSA', 'BNSA'] = 'RNSA'):
        """
        Function responsible for verifying fit function parameters and throwing exceptions if the verification is not successful.

        Parameters:
        ---
             * X (``npt.NDArray``): Training array, containing the samples and their characteristics,
            [``N samples`` (rows)][``N features`` (columns)].
            * y (``npt.NDArray``): Array of target classes of ``X`` with [``N samples`` (lines)].
            * _class_ (Literal[RNSA, BNSA], optional): Current class. Defaults to 'RNSA'.

        ---

        Função responsável por verificar os parâmetros da função fit e lançar exceções se a verificação não for bem-sucedida.

         Parâmetros:
         ---
              * X (``npt.NDArray``): Array de treinamento, contendo as amostras e suas características,
             [``N samples`` (linhas)][``N features`` (colunas)].
             * y (``npt.NDArray``): Array de classes alvo de ``X`` com [``N samples`` (linhas)].
             * _class_ (Literal[RNSA, BNSA], opcional): Classe atual. O padrão é 'RNSA'.
        """
        if not isinstance(X,  (np.ndarray)):
            if isinstance(X,  (list)):
                X = np.array(X)
            else:
                raise TypeError("X is not an ndarray.")
        elif not isinstance(y, (np.ndarray)):
            if isinstance(y, (list)):
                y = np.array(y)
            else:
                raise TypeError("y is not an ndarray.")
        if X.shape[0] != y.shape[0]:
            raise TypeError(
                "X does not have the same amount of sample for the output classes in y.")
        
        if _class_ == 'BNSA' and not np.isin(X, [0, 1]).all():
            raise ValueError(
                "The array X contains values that are not composed only of 0 and 1.")

    def _slice_index_list_by_class(self, y: npt.NDArray) -> dict:
        """
        The function ``__slice_index_list_by_class(...)``, separates the indices of the lines according to the output class,
        to loop through the sample array, only in positions where the output is the class being trained.

        Parameters:
        ---
            * y (npt.NDArray): Receives a ``y``[``N sample``] array with the output classes of the ``X`` sample array.

        returns:
        ---
            * dict: A dictionary with the list of array positions(``y``), with the classes as key.

        ---

        A função ``__slice_index_list_by_class(...)``, separa os índices das linhas conforme a classe de saída, 
        para percorrer o array de amostra, apenas nas posições que a saída for a classe que está sendo treinada.

        Parameters:
        ---
            * y (npt.NDArray): Recebe um array ``y``[``N amostra``] com as classes de saida do array de amostra ``X``.

        Returns:
        ---
            * dict: Um dicionário com a lista de posições do array(``y``), com as classes como chave.
        """
        positionSamples = dict()
        for _class_ in self.classes:
            # Pega as posições das amostras por classes a partir do y.
            positionSamples[_class_] = list(np.where(y == _class_)[0])

        return positionSamples
    
    def _score(self, X: npt.NDArray, y: list) -> float:
        """
        Score function calculates forecast accuracy.

        Details:
        ---
        This function performs the prediction of X and checks how many elements are equal between vector y and y_predicted. 
        This function was added for compatibility with some scikit-learn functions.

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
        Esta função realiza a previsão de X e verifica quantos elementos são iguais entre o vetor y e y_previsto. 
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
        if len(y) == 0:
            return 0
        y_pred = self.predict(X)
        return np.sum(y == y_pred)/len(y)