"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

from unittest.mock import patch

from src.utils import select_saved_data


def test_select_saved_data_empty_list():
    # act
    result = select_saved_data([])

    # assert
    assert result is None


def test_select_saved_data_valid_selection(capsys):
    # arrange
    mock_files = ['data1.pckl', 'data2.pckl', 'data3.pckl']

    # mock and act
    with patch('builtins.input', return_value='1'):
        result = select_saved_data(mock_files)

    # capture and assert
    captured = capsys.readouterr()
    assert result == 'data1.pckl'
    assert 'Saved data data1.pckl selected' in captured.out


def test_select_saved_data_invalid_input(capsys):
    # arrange
    mock_files = ['data1.pckl', 'data2.pckl', 'data3.pckl']

    # mock and act
    with patch('builtins.input', return_value='abc'):
        result = select_saved_data(mock_files)

    # capture and assert
    captured = capsys.readouterr()
    assert result is None
    assert 'Invalid input' in captured.out


def test_select_saved_data_index_out_of_range(capsys):
    # arrange
    mock_files = ['data1.pckl', 'data2.pckl', 'data3.pckl']

    # mock and act
    with patch('builtins.input', return_value='5'):
        result = select_saved_data(mock_files)

    # capture and assert
    captured = capsys.readouterr()
    assert result is None
    assert 'Saved data with number 5 does not exist' in captured.out


def test_select_saved_data_invalid_selection(capsys):
    # arrange
    mock_files = ['data1.pckl', '', 'data3.pckl']

    # mock and act
    with patch('builtins.input', return_value='2'):
        result = select_saved_data(mock_files)

    # capture and assert
    captured = capsys.readouterr()
    assert result is None
    assert 'No valid selection made' in captured.out
