"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

import pickle
from unittest.mock import mock_open, patch

import numpy as np

from src.utils import save_data


def test_save_data_success(tmp_path):
    # arrange
    file_name = tmp_path / 'test_data.pkl'
    array1 = np.array([1, 2, 3])
    array2 = np.array([[1, 2], [3, 4]])

    # act
    save_data(file_name, array1, array2)

    # assert
    with open(file_name, 'rb') as f:
        loaded_data = pickle.load(f)
    msg = 'Loaded data tuple length mismatch'
    assert len(loaded_data) == 2, msg
    msg = 'First array does not match'
    assert np.array_equal(loaded_data[0], array1), msg
    msg = 'Second array does not match'
    assert np.array_equal(loaded_data[1], array2), msg


def test_save_data_ioerror(tmp_path):
    # arrange
    file_name = 'dummy.pkl'
    array1 = np.array([1, 2, 3])

    # mock open to raise an IOError
    with patch('builtins.open', mock_open()) as mocked_open:
        mocked_open.side_effect = IOError('Mocked IO error')

        # act and assert
        msg = (
            'IO error occurred while saving dummy.pkl data file: '
            'Mocked IO error'
        )
        with patch('builtins.print') as mocked_print:
            save_data(file_name, array1)
            mocked_print.assert_any_call(msg)


def test_save_data_generic_exception(tmp_path):
    # arrange
    file_name = 'dummy.pkl'
    array1 = np.array([1, 2, 3])

    # mock pickle.dump to raise a generic exception
    with patch(
        'builtins.open', mock_open()
    ), patch(
        'pickle.dump', side_effect=Exception('Mocked generic error')
    ):
        msg = (
            'An error occurred while saving dummy.pkl data file: '
            'Mocked generic error\n'
        )
        with patch('builtins.print') as mocked_print:
            save_data(file_name, array1)
            mocked_print.assert_any_call(msg)
