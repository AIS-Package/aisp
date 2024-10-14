import numpy as np
from aisp.utils import slice_index_list_by_class


def test_slice_index_list_by_class():
    classes = ['a', 'b']
    y = np.array(['a', 'b', 'a', 'a'])
    expected = {
        'a': [0, 2, 3],
        'b': [1]
    }
    result = slice_index_list_by_class(classes, y)
    assert result == expected
