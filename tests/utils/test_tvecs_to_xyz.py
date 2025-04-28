"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

import numpy as np
import pytest

from src.utils import tvecs_to_xyz


def test_tvecs_to_xyz_column_vector():
    # arrange
    tvecs = np.array([[1], [2], [3]])

    # act
    result = tvecs_to_xyz(tvecs)

    # assert
    assert np.array_equal(result, np.array([1, 2, 3]))


def test_tvecs_to_xyz_1d_array():
    # arrange
    tvecs = np.array([1, 2, 3])

    # act
    result = tvecs_to_xyz(tvecs)

    # assert
    assert np.array_equal(result, np.array([1, 2, 3]))


def test_tvecs_to_xyz_invalid_shape():
    # arrange
    tvecs = np.array([[1], [2]])

    # act and assert
    with pytest.raises(ValueError):
        tvecs_to_xyz(tvecs)

    # arrange
    tvecs = np.array([1, 2])

    # act and assert
    with pytest.raises(ValueError):
        tvecs_to_xyz(tvecs)
