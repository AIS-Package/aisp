from typing import Union
import numpy as np
import numpy.typing as npt


def slice_index_list_by_class(classes: Union[npt.NDArray, list], y: npt.NDArray) -> dict:
    """
    The function ``__slice_index_list_by_class(...)``, separates the indices of the lines \
    according to the output class, to loop through the sample array, only in positions where \
    the output is the class being trained.

    Parameters:
    ---
        * classes (``list or npt.NDArray``): list with unique classes.
        * y (npt.NDArray): Receives a ``y``[``N sample``] array with the output classes of the \
            ``X`` sample array.

    returns:
    ---
        * dict: A dictionary with the list of array positions(``y``), with the classes as key.

    ---

    A função ``__slice_index_list_by_class(...)``, separa os índices das linhas conforme a \
    classe de saída, para percorrer o array de amostra, apenas nas posições que a saída for \
    a classe que está sendo treinada.

    Parameters:
    ---
        * classes (``list or npt.NDArray``): lista com classes únicas.
        * y (npt.NDArray): Recebe um array ``y``[``N amostra``] com as classes de saída do \
            array de amostra ``X``.

    Returns:
    ---
        * dict: Um dicionário com a lista de posições do array(``y``), com as classes como chave.
    """
    position_samples = dict()
    for _class_ in classes:
        # Gets the sample positions by class from y.
        position_samples[_class_] = list(np.nonzero(y == _class_)[0])

    return position_samples
