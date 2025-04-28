"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

from unittest.mock import patch

import numpy as np

from src.calibration import calibrate


def test_calibrate_success(
    sample_calibration_set,
    sample_camera_matrix,
    sample_distortion_matrix,
    sample_vectors,
):
    # arrange
    (sample_obj_points,
     sample_image_points, sample_image_size) = sample_calibration_set
    mock_ret = True
    mock_cam_mtx = sample_camera_matrix
    mock_dist_coeffs = sample_distortion_matrix
    mock_tvecs, mock_rvecs = sample_vectors

    # act
    with patch('cv2.calibrateCamera', return_value=(
        mock_ret, mock_cam_mtx, mock_dist_coeffs, mock_rvecs[0], mock_tvecs[0]
    )) as mock_calibrate:
        ret, cam_mtx, dist_coeffs, rvecs, tvecs = calibrate(
            sample_obj_points, sample_image_points, sample_image_size,
        )

    # assert
    mock_calibrate.assert_called_once_with(
        objectPoints=sample_obj_points,
        imagePoints=sample_image_points,
        imageSize=sample_image_size,
        cameraMatrix=None,
        distCoeffs=None
    )
    assert ret is True
    assert np.array_equal(cam_mtx, mock_cam_mtx)
    assert np.array_equal(dist_coeffs, mock_dist_coeffs)
    assert len(rvecs) == len(sample_obj_points)
    assert len(tvecs) == len(sample_obj_points)


def test_calibrate_no_image_points():
    # arrange
    object_points, image_points, image_size = [], [], []

    # act
    with patch('cv2.calibrateCamera') as mock_calibrate:
        ret, cam_mtx, dist_coeffs, rvecs, tvecs = calibrate(
            object_points, image_points, image_size
        )

    # assert
    mock_calibrate.assert_not_called()
    assert ret is False
    assert cam_mtx.size == 0
    assert dist_coeffs.size == 0
    assert rvecs == ()
    assert tvecs == ()


def test_calibrate_partial_inputs(sample_calibration_set):
    # arrange
    sample_obj_points, _, sample_image_size = sample_calibration_set
    sample_image_points = []

    # act
    with patch('cv2.calibrateCamera') as mock_calibrate:
        ret, cam_mtx, dist_coeffs, rvecs, tvecs = calibrate(
            sample_obj_points, sample_image_points, sample_image_size,
        )

    # assert
    mock_calibrate.assert_not_called()
    assert ret is False
    assert cam_mtx.size == 0
    assert dist_coeffs.size == 0
    assert rvecs == ()
    assert tvecs == ()
