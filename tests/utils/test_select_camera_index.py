"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

from unittest.mock import patch

from src.utils import select_camera_index


def test_select_camera_index_valid_input():
    # arrange
    with patch(
        'builtins.input', side_effect=['2']
    ), patch('builtins.print') as mock_print:

        # act
        result = select_camera_index()

        # assert
        msg = 'Expected camera index to return 2 for input "2"'
        assert result == 2, msg
        mock_print.assert_any_call('\nSelected camera index 2\n')


def test_select_camera_index_invalid_then_valid_input():
    # arrange
    with patch(
        'builtins.input', side_effect=['a', '3']
    ), patch('builtins.print') as mock_print:

        # act
        result = select_camera_index()

        # assert
        msg = 'Expected camera index to return 3 after initial invalid input'
        assert result == 3, msg
        mock_print.assert_any_call('\nInvalid input\n')
        mock_print.assert_any_call('\nSelected camera index 3\n')


def test_select_camera_index_multiple_invalid_then_valid_input():
    # arrange
    with patch(
        'builtins.input', side_effect=['x', 'y', '1']
    ), patch('builtins.print') as mock_print:

        # act
        result = select_camera_index()

        # assert
        msg = 'Expected camera index to return 1 after multiple invalid inputs'
        assert result == 1, msg
        msg = 'Expected multiple prompts for invalid inputs'
        assert mock_print.call_count >= 3, msg
        mock_print.assert_any_call('\nInvalid input\n')
        mock_print.assert_any_call('\nSelected camera index 1\n')


def test_select_camera_index_edge_case_large_number():
    # arrange
    with patch(
        'builtins.input', side_effect=['100']
    ), patch('builtins.print') as mock_print:

        # act
        result = select_camera_index()

        # assert
        msg = 'Expected camera index to return 100 for input "100"'
        assert result == 100, msg
        mock_print.assert_any_call('\nSelected camera index 100\n')
