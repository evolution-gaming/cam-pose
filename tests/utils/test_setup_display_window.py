"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

from unittest.mock import patch

import cv2 as cv

from src.utils import setup_display_window


def test_setup_display_window(sample_settings):
    # act and assert
    with patch('cv2.namedWindow') as mock_named_window, \
         patch('cv2.resizeWindow') as mock_resize_window:

        setup_display_window(
            sample_settings.CB_PATT,
            sample_settings.WINDOW_WIDTH,
            sample_settings.WINDOW_HEIGHT,
        )

        mock_named_window.assert_called_once_with(
            sample_settings.CB_PATT, cv.WINDOW_NORMAL,
        )
        mock_resize_window.assert_called_once_with(
            sample_settings.CB_PATT,
            sample_settings.WINDOW_WIDTH,
            sample_settings.WINDOW_HEIGHT,
        )
