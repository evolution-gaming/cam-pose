"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

import cv2 as cv
import numpy as np
import pytest

from src.utils import add_menu


def test_add_menu_positions_text_correctly(sample_image, sample_settings):
    # arrange
    menu = {'A': 'Additional Info', 'C': 'Corners', 'F': 'Flip Image'}
    color = sample_settings.WHITE
    line_height = 30

    # act
    add_menu(sample_image, menu, color)

    # assert
    for i, (k, v) in enumerate(menu.items()):
        temp_image = np.zeros_like(sample_image)
        cv.putText(
            temp_image,
            f'[{k}] {v}',
            (sample_image.shape[1] - 200,
             sample_image.shape[0]
             - (len(menu) * line_height)
             + i
             * line_height),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
        )
        msg = 'Text is not positioned correctly.'
        assert np.any(cv.bitwise_and(temp_image, sample_image)), msg


def test_add_menu_does_not_crash_with_empty_menu(
    sample_image, sample_settings,
):
    # arrange
    menu = {}
    color = sample_settings.WHITE

    # act
    try:
        add_menu(sample_image, menu, color)
    except Exception as e:
        pytest.fail(f'add_menu crashed with an empty menu: {e}')

    # assert
    msg = 'Image should remain unchanged when menu is empty.'
    assert np.array_equal(sample_image,
                          np.zeros((500, 500, 3), dtype=np.uint8)), msg


def test_add_menu_correct_color(sample_image, sample_settings):
    # arrange
    menu = {'A': 'Additional Info', 'C': 'Corners'}
    color = sample_settings.GREEN

    # act
    add_menu(sample_image, menu, color)

    # assert
    msg = 'Text is not rendered in the correct color (Green).'
    assert np.any((
        sample_image[:, :, 0] == 0)
        & (sample_image[:, :, 1] == 255)
        & (sample_image[:, :, 2] == 0)), msg
