"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

from unittest.mock import call, patch

import cv2 as cv

from src.utils import add_text_xyz


def test_add_text_xyz_standard_case(
    sample_image, sample_distances, sample_angles, sample_settings,
):
    # arrange
    title = 'Title'
    expected_calls = [
        call(
            sample_image,
            title,
            (60, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        ),
        call(
            sample_image,
            'x=1.23 mm, r=10.11 deg',
            (60, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        ),
        call(
            sample_image,
            'y=4.56 mm, p=20.22 deg',
            (60, 90),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        ),
        call(
            sample_image,
            'z=7.89 mm, y=30.33 deg',
            (60, 120),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
    ]

    # act and assert
    with patch('cv2.putText') as mock_putText:
        add_text_xyz(
            sample_image,
            sample_distances,
            sample_angles,
            sample_settings,
            title,
            30,
            30
        )
        mock_putText.assert_has_calls(expected_calls, any_order=False)


def test_add_text_xyz_with_nan_values(
    sample_image, sample_distances_nan, sample_angles_nan, sample_settings,
):
    # arrange
    title = 'Title'
    expected_calls = [
        call(
            sample_image,
            title,
            (60, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        ),
        call(
            sample_image,
            'x=NA mm, r=10.11 deg',
            (60, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        ),
        call(
            sample_image,
            'y=4.56 mm, p=NA deg',
            (60, 90),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        ),
        call(
            sample_image,
            'z=7.89 mm, y=30.33 deg',
            (60, 120),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
    ]

    # act and assert
    with patch('cv2.putText') as mock_putText:
        add_text_xyz(
            sample_image,
            sample_distances_nan,
            sample_angles_nan,
            sample_settings,
            title,
            30,
            30
        )
        mock_putText.assert_has_calls(expected_calls, any_order=False)


def test_add_text_xyz_color_and_position(
    sample_image, sample_distances, sample_angles, sample_settings,
):
    # arrange
    title = 'Test Title'
    expected_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    # act and assert
    with patch('cv2.putText') as mock_putText:
        add_text_xyz(
            sample_image,
            sample_distances,
            sample_angles,
            sample_settings,
            title,
            50,
            40
        )
        for i, color in enumerate(expected_colors):
            assert mock_putText.call_args_list[i + 1][0][5] == color


def test_add_text_xyz_with_empty_arrays(
    sample_image, sample_nan_all, sample_settings,
):
    # act and assert
    with patch('cv2.putText') as mock_putText:
        add_text_xyz(
            sample_image,
            sample_nan_all,
            sample_nan_all,
            sample_settings,
            'Title',
            30,
            30,
        )
        mock_putText.assert_any_call(
            sample_image,
            'x=NA mm, r=NA deg',
            (60, 60),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )
        mock_putText.assert_any_call(
            sample_image,
            'y=NA mm, p=NA deg',
            (60, 90),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        mock_putText.assert_any_call(
            sample_image,
            'z=NA mm, y=NA deg',
            (60, 120),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
