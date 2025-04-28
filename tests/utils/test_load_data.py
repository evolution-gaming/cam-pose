"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

import pickle
from unittest.mock import mock_open, patch

from src.utils import load_data


def test_load_data_success():
    # arrange
    sample_data = (1, 2, 3)
    pickled_data = pickle.dumps(sample_data)

    # mock opening a file and returning pickled data
    with patch('builtins.open', mock_open(read_data=pickled_data)):

        # act
        result = load_data('dummy_file.pkl')

        # assert
        msg = 'Expected the data to be loaded successfully'
        assert result == sample_data, msg


def test_load_data_file_not_found(capfd):
    # arrange
    with patch('builtins.open', side_effect=FileNotFoundError):

        # act
        result = load_data('non_existent_file.pkl')

        # capture printed output
        captured = capfd.readouterr()

        # assert
        msg = 'Expected empty tuple when file is not found'
        assert result == (), msg
        msg = 'File non_existent_file.pkl not found'
        assert msg in captured.out


def test_load_data_unpickling_error(capfd):
    # arrange
    file_name = 'corrupted_file.pkl'
    with patch('builtins.open', mock_open(read_data='invalid data')), patch(
        'pickle.load', side_effect=pickle.UnpicklingError
    ):

        # act
        result = load_data(file_name)

        # capture printed output
        captured = capfd.readouterr()

        # assert
        msg = 'Expected empty tuple when file contains invalid pickle data'
        assert result == (), msg
        msg = f'The file {file_name} may not be a valid pickle file'
        assert msg in captured.out


def test_load_data_generic_exception(capfd):
    # arrange
    with patch('builtins.open', mock_open(read_data='data')), patch(
        'pickle.load', side_effect=Exception('Unexpected error')
    ):
        # act
        result = load_data('error_file.pkl')

        # capture printed output
        captured = capfd.readouterr()

        # assert
        msg = 'Expected empty tuple on generic exception'
        assert result == (), msg
        msg = 'An error occurred while loading data: Unexpected error\n\n'
        assert msg in captured.out
