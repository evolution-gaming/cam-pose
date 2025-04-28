"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

import numpy as np

from src.utils import generate_3d_axis


def test_generate_3d_axis_positive_size():
    # arrange
    axis_size = 10.0

    # act
    axis = generate_3d_axis(axis_size)

    # assert: shape
    expected_shape = (3, 3)
    msg = f'Expected shape {expected_shape}, got {axis.shape}'
    assert axis.shape == expected_shape, msg

    # assert: values
    expected_values = np.float32([
        [axis_size, 0, 0],
        [0, axis_size, 0],
        [0, 0, -axis_size]
    ])
    msg = 'Generated 3D axis values are incorrect'
    assert np.allclose(axis, expected_values), msg


def test_generate_3d_axis_zero_size():
    # arrange
    axis_size = 0.0

    # act
    axis = generate_3d_axis(axis_size)

    # assert
    expected_values = np.float32([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ])
    msg = 'Expected all zero values with axis size 0'
    assert np.allclose(axis, expected_values), msg


def test_generate_3d_axis_negative_size():
    # arrange
    axis_size = -5.0

    # act
    axis = generate_3d_axis(axis_size)

    # assert
    expected_values = np.float32([
        [axis_size, 0, 0],
        [0, axis_size, 0],
        [0, 0, -axis_size]
    ])
    msg = 'Generated 3D axis values do not match for negative size'
    assert np.allclose(axis, expected_values), msg
