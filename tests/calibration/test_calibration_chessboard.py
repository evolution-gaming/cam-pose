"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

from unittest import mock

import numpy as np

from src.calibration import calibration_chessboard


def test_calibration_success(
    sample_calibration_set, sample_video_capture, sample_settings,
):
    # mock
    with mock.patch(
            'cv2.cvtColor',
            return_value=np.zeros((500, 500), dtype=np.uint8)
        ), \
        mock.patch('cv2.VideoCapture'), \
        mock.patch('cv2.drawChessboardCorners'), \
        mock.patch(
            'cv2.findChessboardCorners',
            return_value=(True, np.zeros((35, 1, 2), dtype=np.float32))
        ), \
        mock.patch(
            'cv2.cornerSubPix',
            return_value=np.zeros((35, 1, 2), dtype=np.float32)
        ), \
        mock.patch(
            'cv2.calibrateCamera',
            return_value=(
                True,
                'mock_cam_mtx',
                'mock_dist_coeffs',
                'mock_rvecs',
                'mock_tvecs'
            )) as mock_calibratecamera, \
        mock.patch('cv2.imshow'), \
        mock.patch(
            'cv2.waitKey',
            side_effect=[-1, 27]
        ) as mock_waitkey, \
        mock.patch(
            'cv2.imwrite'
    ):

        # arrange
        (
            sample_obj_points, sample_img_points, sample_img_size,
        ) = sample_calibration_set

        # act
        (
            ret, cam_mtx, dist_coeffs, rvecs, tvecs, obj_points, img_points
        ) = calibration_chessboard(
            video_capture=sample_video_capture,
            object_points=sample_obj_points,
            image_points=sample_img_points,
            img_size=sample_img_size,
            configuration=sample_settings
        )

        # assert
        msg = 'Expected calibration to succeed'
        assert ret is True, msg
        msg = 'Expected camera matrix from calibration'
        assert cam_mtx == 'mock_cam_mtx', msg
        msg = 'Expected five object points entry'
        assert len(sample_obj_points) == 5, msg
        msg = 'Expected five image points entry'
        assert len(sample_img_points) == 5, msg
        msg = 'Expected cv.calibrateCamera to be called'
        assert mock_calibratecamera.called, msg
        assert mock_waitkey.call_count == 2
