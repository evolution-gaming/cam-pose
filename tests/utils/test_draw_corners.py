"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

from unittest.mock import patch

import numpy as np

from src.utils import draw_corners


def test_draw_corners_on_image(sample_settings):
    # arrange
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    corners = np.array([[[10, 10]], [[20, 20]], [[30, 30]]], dtype=np.float32)

    # act and assert
    with patch('cv2.circle') as mock_circle:
        draw_corners(image, corners, sample_settings.BLUE)

        assert mock_circle.call_count == len(corners)

        for _, corner in enumerate(corners):
            x, y = map(int, corner.ravel())
            mock_circle.assert_any_call(
                image, (x, y), 5, sample_settings.BLUE, -1,
            )


def test_draw_corners_no_corners(sample_settings):
    # arrange
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    corners = np.array([], dtype=np.float32).reshape(0, 1, 2)

    # act and assert
    with patch('cv2.circle') as mock_circle:
        draw_corners(image, corners, sample_settings.GREEN)

        assert mock_circle.call_count == 0


def test_draw_corners_color(sample_settings):
    # arrange
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    corners = np.array([[[50, 50]]], dtype=np.float32)

    # act
    draw_corners(image, corners, sample_settings.RED)

    # assert
    msg = 'The color at the corner does not match the specified color.'
    assert np.array_equal(image[50, 50], sample_settings.RED), msg


def test_draw_corners_out_of_bounds(sample_settings):
    # arrange
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    corners = np.array([[[150, 150]]], dtype=np.float32)

    # act and assert
    with patch('cv2.circle') as mock_circle:
        draw_corners(image, corners, sample_settings.GREEN)

        mock_circle.assert_called_once()
