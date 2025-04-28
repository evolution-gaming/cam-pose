"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

from unittest.mock import patch

from src.utils import generate_file_name


def test_generate_file_name_basic():
    # arrange
    title = 'sample'
    extension = 'jpg'
    pattern = 'chessboard'

    # patch datetime
    with patch('src.utils.datetime') as mock_dt:
        mock_dt.now.return_value.strftime.return_value = '2023_10_10__12_00_00'

        # act
        result = generate_file_name(title, extension, pattern)

    # assert
    expected = 'sample_jpg_of_chessboard_2023_10_10__12_00_00.jpg'
    assert result == expected, f'Expected "{expected}", but got "{result}"'


def test_generate_file_name_empty_title():
    # arrange
    title = ''
    extension = 'jpg'
    pattern = 'pattern'

    # patch datetime to control the output
    with patch('src.utils.datetime') as mock_dt:
        mock_dt.now.return_value.strftime.return_value = '2023_10_10__12_00_00'

        # act
        result = generate_file_name(title, extension, pattern)

    # assert
    expected = '_jpg_of_pattern_2023_10_10__12_00_00.jpg'
    assert result == expected, f'Expected "{expected}"", but got "{result}"'


def test_generate_file_name_empty_pattern():
    # arrange
    title = 'title'
    extension = 'jpg'
    pattern = ''

    # patch datetime to control the output
    with patch('src.utils.datetime') as mock_dt:
        mock_dt.now.return_value.strftime.return_value = '2023_10_10__12_00_00'

        # act
        result = generate_file_name(title, extension, pattern)

    # assert
    expected = 'title_jpg_of__2023_10_10__12_00_00.jpg'
    assert result == expected, f'Expected "{expected}", but got "{result}"'


def test_generate_file_name_empty_extension():
    # arrange
    title = 'title'
    extension = ''
    pattern = 'pattern'

    # patch datetime to control the output
    with patch('src.utils.datetime') as mock_dt:
        mock_dt.now.return_value.strftime.return_value = '2023_10_10__12_00_00'

        # act
        result = generate_file_name(title, extension, pattern)

    # assert
    expected = 'title__of_pattern_2023_10_10__12_00_00.'
    assert result == expected, f'Expected "{expected}", but got "{result}"'


def test_generate_file_name_empty_title_extension_pattern():
    # arrange
    title = ''
    extension = ''
    pattern = ''

    # patch datetime to control the output
    with patch('src.utils.datetime') as mock_dt:
        mock_dt.now.return_value.strftime.return_value = '2023_10_10__12_00_00'

        # act
        result = generate_file_name(title, extension, pattern)

    # assert
    expected = '__of__2023_10_10__12_00_00.'
    assert result == expected, f'Expected "{expected}"", but got "{result}"'
