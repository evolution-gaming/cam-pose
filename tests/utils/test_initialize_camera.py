"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

import platform
from unittest.mock import MagicMock, patch

import cv2 as cv

from src.utils import initialize_camera


@patch('cv2.VideoCapture')
def test_initialize_camera_success(mock_VideoCapture, sample_settings):
    # arrange
    camera_index = 0
    mock_capture = MagicMock()
    mock_VideoCapture.return_value = mock_capture

    # act
    capture = initialize_camera(camera_index, sample_settings)

    # assert
    if platform.system() == 'Windows':
        mock_VideoCapture.assert_called_once_with(camera_index, cv.CAP_DSHOW)
    else:
        mock_VideoCapture.assert_called_once_with(camera_index)
    mock_capture.set.assert_any_call(
        cv.CAP_PROP_FRAME_WIDTH, sample_settings.FRAME_WIDTH,
    )
    mock_capture.set.assert_any_call(
        cv.CAP_PROP_FRAME_HEIGHT, sample_settings.FRAME_HEIGHT,
    )
    assert capture == mock_capture


@patch('cv2.VideoCapture')
def test_initialize_camera_failure(mock_VideoCapture, sample_settings):
    # arrange
    camera_index = 0
    mock_capture = MagicMock()
    mock_capture.isOpened.return_value = False
    mock_VideoCapture.return_value = mock_capture

    # act
    capture = initialize_camera(camera_index, sample_settings)

    # assert
    if platform.system() == 'Windows':
        mock_VideoCapture.assert_called_once_with(camera_index, cv.CAP_DSHOW)
    else:
        mock_VideoCapture.assert_called_once_with(camera_index)
    mock_capture.isOpened.assert_called_once()
    assert capture is None
