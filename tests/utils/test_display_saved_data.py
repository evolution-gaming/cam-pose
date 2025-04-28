"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

from src.utils import display_saved_data


def test_display_saved_data_files_found(capsys):
    # mock a directory with .pckl files
    mock_files = [
        MagicMock(spec=Path, name='file1.pckl'),
        MagicMock(spec=Path, name='file2.pckl')
    ]
    for mock_file in mock_files:
        mock_file.is_file.return_value = True
        mock_file.suffix = '.pckl'
        mock_file.name = mock_file._mock_name

    with patch('src.utils.Path.iterdir', return_value=mock_files):
        result = display_saved_data(Path.cwd())

        # capture and assert output
        captured = capsys.readouterr()
        assert 'Available saved data:' in captured.out
        assert '1 - file1.pckl' in captured.out
        assert '2 - file2.pckl' in captured.out
        assert result == ['file1.pckl', 'file2.pckl']


def test_display_saved_data_no_files(capsys):
    # mock an empty directory
    with patch('src.utils.Path.iterdir', return_value=[]):
        result = display_saved_data(Path.cwd())

        # capture and assert output
        captured = capsys.readouterr()
        assert 'No saved data found' in captured.out
        assert result == []


def test_display_saved_data_oserror(capsys):
    # mock a directory access error
    with patch('src.utils.Path.iterdir', side_effect=OSError('Access error')):
        result = display_saved_data(Path.cwd())

        # capture and assert output
        captured = capsys.readouterr()
        assert 'Error accessing directory' in captured.out
        assert result == []
