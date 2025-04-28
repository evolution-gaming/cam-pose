"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

from unittest.mock import call, patch

import numpy as np

from src.utils import draw_axis_3d


@patch('cv2.line')
def test_draw_axis_3d_calls_cv_line(
    mock_cv_line,
    sample_image,
    sample_corners,
    sample_image_points,
    sample_settings,
):
    # arrange
    corner = tuple(sample_corners.ravel().astype(int))
    expected_calls = [
        call(sample_image, corner, (60, 50), sample_settings.BLUE, 5),
        call(sample_image, corner, (50, 60), sample_settings.GREEN, 5),
        call(sample_image, corner, (50, 40), sample_settings.RED, 5)
    ]

    # act
    draw_axis_3d(
        sample_image, sample_corners, sample_image_points, sample_settings,
    )

    # assert
    mock_cv_line.assert_has_calls(expected_calls, any_order=False)


def test_draw_axis_3d_changes_image(
    sample_image, sample_corners, sample_image_points, sample_settings,
):
    # arrange
    modified_image = draw_axis_3d(
        sample_image.copy(),
        sample_corners,
        sample_image_points,
        sample_settings,
    )

    # act and assert
    assert np.array_equal(modified_image[50, 60], sample_settings.BLUE)
    assert np.array_equal(modified_image[60, 50], sample_settings.GREEN)
    assert np.array_equal(modified_image[40, 50], sample_settings.RED)


def test_draw_axis_3d_with_empty_image_points(
    sample_image, sample_corners, sample_settings,
):
    # arrange
    empty_image_points = np.empty((0, 1, 2), dtype=np.float32)

    # act
    result_image = draw_axis_3d(
        sample_image.copy(),
        sample_corners,
        empty_image_points,
        sample_settings,
    )

    # assert
    assert np.array_equal(result_image, sample_image)
