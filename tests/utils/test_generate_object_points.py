"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

import numpy as np

from src.utils import generate_object_points


def test_generate_object_points_shape(sample_settings):
    # act
    obj_points = generate_object_points(sample_settings)

    # assert
    expected_shape = (
        sample_settings.CB_ROWS * sample_settings.CB_COLUMNS, 3
    )
    msg = f'Expected shape {expected_shape}, got {obj_points.shape}'
    assert obj_points.shape == expected_shape, msg


def test_generate_object_points_values(sample_settings):
    # act
    obj_points = generate_object_points(sample_settings)

    # assert
    expected_values = np.zeros(
        (sample_settings.CB_ROWS * sample_settings.CB_COLUMNS, 3), np.float32
    )
    expected_values[:, :2] = (
        np.mgrid[0:sample_settings.CB_ROWS, 0:sample_settings.CB_COLUMNS]
        .T.reshape(-1, 2) * sample_settings.SQUARE_SIZE
    )
    msg = 'Object points values are incorrect'
    assert np.allclose(obj_points, expected_values), msg


def test_generate_object_points_zero_square_size():
    # arrange
    class ZeroSquareSettings:
        CB_ROWS = 5
        CB_COLUMNS = 5
        SQUARE_SIZE = 0.0

    # act
    obj_points = generate_object_points(ZeroSquareSettings)

    # assert
    expected_shape = (
        ZeroSquareSettings.CB_ROWS * ZeroSquareSettings.CB_COLUMNS, 3
    )
    msg = f'Expected shape {expected_shape}, got {obj_points.shape}'
    assert obj_points.shape == expected_shape, msg
    msg = 'Expected all coordinates to be zero with SQUARE_SIZE of 0'
    assert np.all(obj_points[:, :2] == 0), msg
