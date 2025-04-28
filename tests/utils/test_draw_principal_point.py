"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

import numpy as np
import pytest

from src.utils import draw_principal_point


def test_draw_principal_point_with_valid_camera_matrix(
    sample_image, sample_camera_matrix,
):
    # act
    draw_principal_point(sample_image, sample_camera_matrix)

    # assert
    assert sample_image[250, 250].tolist() == [0, 0, 255]
    assert sample_image[249, 250].tolist() == [0, 0, 255]
    assert sample_image[251, 250].tolist() == [0, 0, 255]
    assert sample_image[250, 249].tolist() == [0, 0, 255]
    assert sample_image[250, 251].tolist() == [0, 0, 255]


def test_draw_principal_point_with_different_color(
    sample_image, sample_camera_matrix,
):
    # act
    draw_principal_point(sample_image, sample_camera_matrix, color=(0, 255, 0))

    # assert
    assert sample_image[250, 250].tolist() == [0, 255, 0]
    assert sample_image[249, 250].tolist() == [0, 255, 0]
    assert sample_image[251, 250].tolist() == [0, 255, 0]
    assert sample_image[250, 249].tolist() == [0, 255, 0]
    assert sample_image[250, 251].tolist() == [0, 255, 0]


def test_invalid_camera_matrix(sample_image):
    # arrange
    invalid_camera_matrix = np.array([
        [1, 0, 250],
        [0, 1, 250]
    ], dtype=np.float32)

    # act and assert
    msg = 'Expected camera matrix to be a 3x3 array.'
    with pytest.raises(ValueError, match=msg):
        draw_principal_point(sample_image, invalid_camera_matrix)
