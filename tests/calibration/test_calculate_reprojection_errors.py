"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

import numpy as np
import pytest

from src.calibration import calculate_reprojection_errors


def test_basic_reprojection_error(
    sample_camera_matrix,
    sample_data_points,
    sample_distortion_matrix,
    sample_vectors,
):
    # arrange
    sample_obj_p, sample_img_p = sample_data_points
    sample_tvecs, sample_rvecs = sample_vectors

    # act
    errors = calculate_reprojection_errors(
        sample_obj_p,
        sample_rvecs,
        sample_tvecs,
        sample_camera_matrix,
        sample_distortion_matrix,
        sample_img_p,
    )

    # assert
    assert len(errors) == 1
    assert errors[0] >= 0


def test_mismatched_lengths(
    sample_camera_matrix,
    sample_data_points,
    sample_distortion_matrix,
    sample_vectors,
):
    # arrange
    sample_obj_p, sample_img_p = sample_data_points
    sample_tvecs, sample_rvecs = sample_vectors

    # act and assert
    msg = 'All input lists must have the same length'
    with pytest.raises(ValueError, match=msg):
        calculate_reprojection_errors(
            sample_obj_p,
            sample_rvecs * 2,
            sample_tvecs,
            sample_camera_matrix,
            sample_distortion_matrix,
            sample_img_p,
        )


def test_single_point_set(sample_camera_matrix, sample_distortion_matrix):
    # arrange
    object_points = [np.array([[0, 0, 0]], dtype=np.float32)]
    rotation_vectors = (np.array([[0.0], [0.0], [0.0]], dtype=np.float32),)
    translation_vectors = (np.array([[0.0], [0.0], [0.0]], dtype=np.float32),)
    image_points = [np.array([[0, 0]], dtype=np.float32)]

    # act
    errors = calculate_reprojection_errors(
        object_points,
        rotation_vectors,
        translation_vectors,
        sample_camera_matrix,
        sample_distortion_matrix,
        image_points,
    )

    # assert
    assert len(errors) == 1
    assert errors[0] == 353.5533905932738


def test_high_error_scenario(sample_camera_matrix, sample_distortion_matrix):
    # arrange
    object_points = [np.array([[0, 0, 0], [10, 10, 10]], dtype=np.float32)]
    rotation_vectors = (np.array([[0.0], [0.0], [0.0]], dtype=np.float32),)
    translation_vectors = (np.array([[0.0], [0.0], [0.0]], dtype=np.float32),)
    image_points = [np.array([[100, 100], [200, 200]], dtype=np.float32)]

    # act
    errors = calculate_reprojection_errors(
        object_points,
        rotation_vectors,
        translation_vectors,
        sample_camera_matrix,
        sample_distortion_matrix,
        image_points,
    )

    # assert
    assert len(errors) == 1
    msg = 'Expected a high reprojection error'
    assert errors[0] > 10, msg


def test_empty_inputs(sample_camera_matrix, sample_distortion_matrix):
    errors = calculate_reprojection_errors(
        [], (), (), sample_camera_matrix, sample_distortion_matrix, [],
    )
    msg = 'Expected empty error list for empty inputs'
    assert errors == [], msg
