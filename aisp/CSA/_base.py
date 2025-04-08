from abc import abstractmethod

import numpy as np
import numpy.typing as npt
from typing import Literal, Optional
from scipy.spatial.distance import euclidean, cityblock, minkowski

from ..utils.metrics import accuracy_score


class BaseClassifier:
    """
    The base class contains functions that are used by more than one class in the package, and \
    therefore are considered essential for the overall functioning of the system.

    ---

    A classe base contém funções que são utilizadas por mais de uma classe do pacote, e por isso \
    são consideradas essenciais para o funcionamento geral do sistema.
    """

    def __init__(self, metric: str = "euclidean", p: float = 2):
        """
        Parameters:
        ---
        * metric (``str``): Way to calculate the distance between the detector and the sample:

                * ``'Euclidean'`` ➜ The calculation of the distance is given by the expression: \
                    √( (x₁ – x₂)² + (y₁ – y₂)² + ... + (yn – yn)²).
                * ``'minkowski'`` ➜ The calculation of the distance is given by the expression: \
                    ( |X₁ – Y₁|p + |X₂ – Y₂|p + ... + |Xn – Yn|p) ¹/ₚ.
                * ``'manhattan'`` ➜ The calculation of the distance is given by the expression: \
                    ( |x₁ – x₂| + |y₁ – y₂| + ... + |yn – yn|) .


        * p (``float``): This parameter stores the value of ``p`` used in the Minkowski distance.\
            The default is ``2``, which represents normalized Euclidean distance. Different \
            values of p lead to different variants of the Minkowski distance \
            [learn more](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.minkowski.html).

        ---

        Parameters:
        ---
        * metric (``str``): Forma para se calcular a distância entre o detector e amostra:

                * ``'euclidiana'`` ➜ O cálculo da distância dá-se pela expressão: \
                    √( (x₁ – x₂)² + (y₁ – y₂)² + ... + (yn – yn)²).
                * ``'minkowski'``  ➜ O cálculo da distância dá-se pela expressão: \
                    ( |X₁ – Y₁|p + |X₂ – Y₂|p + ... + |Xn – Yn|p).
                * ``'manhattan'``  ➜ O cálculo da distância dá-se pela expressão: \
                    ( |x₁ – x₂| + |y₁ – y₂| + ... + |yn – yn|).

            Defaults to ``'Euclidean'``.

        * p (``float``): Este parâmetro armazena o valor de ``p`` utilizada na distância de Minkowski. \
            O padrão é ``2``, o que significa distância euclidiana normalizada. Diferentes valores \
            de p levam a diferentes variantes da distância de Minkowski \
            [saiba mais](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.minkowski.html).
        """
        if metric == "manhattan" or metric == "minkowski":
            self.metric: str = metric
        else:
            self.metric: str = "euclidean"
        self.p: float = p

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
        if self.metric == "manhattan":
            return cityblock(u, v)
        elif self.metric == "minkowski":
            return minkowski(u, v, self.p)
        else:
            return euclidean(u, v)

    @staticmethod
    def _check_and_raise_exceptions_fit(
            X: npt.NDArray = None,
            y: npt.NDArray = None,
            algorithm: Literal[
                "continuous-features", "binary-features"
            ] = "continuous-features"
    ):
        """
        Function responsible for verifying fit function parameters and throwing exceptions if the \
        verification is not successful.

        Parameters:
        ---
            * X (``npt.NDArray``): Training array, containing the samples and their characteristics, \
                [``N samples`` (rows)][``N features`` (columns)].
            * y (``npt.NDArray``): Array of target classes of ``X`` with [``N samples`` (lines)].
            * algorithm (Literal[RNSA, BNSA], optional): Current class. Defaults to 'RNSA'.

        Raises:
        ---
            * TypeError: If X or y are not ndarrays or have incompatible shapes.
            * ValueError: If _class_ is BNSA and X contains values that are not composed only of 0 and 1.

        ---

        Função responsável por verificar os parâmetros da função fit e lançar exceções se a \
        verificação não for bem-sucedida.

        Parâmetros:
        ---
            * X (``npt.NDArray``): Array de treinamento, contendo as amostras e suas características, \
                [``N samples`` (linhas)][``N features`` (colunas)].
            * y (``npt.NDArray``): Array de classes alvo de ``X`` com [``N samples`` (linhas)].
            * algorithm (Literal[continuous-features, binary-features], opcional): Classe atual. \
                O padrão é 'continuous-features'.

        Lança:
        ---
            * TypeError: Se X ou y não forem ndarrays ou tiverem formas incompatíveis.
            * ValueError: Se _class_ for BNSA e X contiver valores que não sejam compostos apenas por 0 e 1.
        """
        if not isinstance(X, np.ndarray):
            if isinstance(X, list):
                X = np.array(X)
            else:
                raise TypeError("X is not an ndarray or list.")
        elif not isinstance(y, np.ndarray):
            if isinstance(y, list):
                y = np.array(y)
            else:
                raise TypeError("y is not an ndarray or list.")
        if X.shape[0] != y.shape[0]:
            raise TypeError(
                "X does not have the same amount of sample for the output classes in y."
            )

        if algorithm == "binary-features" and not np.isin(X, [0, 1]).all():
            raise ValueError(
                "The array X contains values that are not composed only of 0 and 1."
            )

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
        Esta função realiza a previsão de X e verifica quantos elementos são iguais entre o vetor \
        y e y_previsto. Essa função foi adicionada para oferecer compatibilidade com algumas funções \
        do scikit-learn.

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
        return accuracy_score(y, y_pred)

    @abstractmethod
    def fit(self, X: npt.NDArray, y: npt.NDArray, verbose: bool = True):
        """
        Function to train the model using the input data ``X`` and corresponding labels ``y``.

        This abstract method is implemented by the class that inherits it.

        Parameters:
        ---
            * X (``npt.NDArray``): Input data used for training the model.
            * y (``npt.NDArray``): Corresponding labels or target values for the input data.
            * verbose (``bool``, optional): Flag to enable or disable detailed output during \
                training. Default is ``True``.

        Returns:
        ---
            * self: Returns the instance of the class that implements this method.

        ---

        Função para treinar o modelo usando os dados de entrada ``X`` e os classes correspondentes ``y``.

        Este método abstrato é implementado pela classe que o herdar.

        Parâmetros:
        ---
            * X (``npt.NDArray``): Dados de entrada utilizados para o treinamento do modelo.
            * y (``npt.NDArray``): Rótulos ou valores-alvo correspondentes aos dados de entrada.
            * verbose (``bool``, opcional): Flag para ativar ou desativar a saída detalhada durante o \
                treinamento. O padrão é ``True``.

        Retornos:
        ---
            * self: Retorna a instância da classe que implementa este método.
        """
        pass

    @abstractmethod
    def predict(self, X) -> Optional[npt.NDArray]:
        """
        Function to generate predictions based on the input data ``X``.

        This abstract method is implemented by the class that inherits it.

        Parameters:
        ---
            * X (``npt.NDArray``): Input data for which predictions will be generated.

        Returns:
        ---
            * Predictions (``Optional[npt.NDArray]``): Predicted values for each input sample, or ``None``
                if the prediction fails.

        ---

        Função para gerar previsões com base nos dados de entrada ``X``.

        Este método abstrato é implementado pela classe que o herdar.

        Parâmetros:
        ---
            * X (``npt.NDArray``): Dados de entrada para os quais as previsões serão geradas.

        Retornos:
        ---
            * Previsões (``Optional[npt.NDArray]``): Valores previstos para cada amostra de entrada,
                ou ``None`` se a previsão falhar.
        """
        pass
