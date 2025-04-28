"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

from unittest import mock

import numpy as np

from src.reference import reference_chessboard


def test_reference_chessboard_success(
    sample_3d_axis,
    sample_angles,
    sample_camera_matrix,
    sample_corners,
    sample_distances,
    sample_distortion_matrix,
    sample_video_capture,
    sample_settings,
):
    # mock
    with mock.patch(
            'cv2.cvtColor',
            return_value=np.zeros((500, 500), dtype=np.uint8)
        ) as mock_cvtcolor, \
        mock.patch(
            'cv2.findChessboardCorners',
            return_value=(True, np.zeros((35, 1, 2), dtype=np.float32))
        ) as mock_find_corners, \
        mock.patch(
            'cv2.cornerSubPix',
            return_value=np.zeros((35, 1, 2), dtype=np.float32)
        ) as mock_subpix, \
        mock.patch('cv2.drawChessboardCorners') as mock_drawtemplatecorners, \
        mock.patch(
            'cv2.solvePnP',
            return_value=(True, np.zeros((3, 1)), np.zeros((3, 1)))
        ) as mock_solvepnp, \
        mock.patch(
            'cv2.projectPoints',
            return_value=(np.zeros((4, 2), dtype=np.float32), None)
        ) as mock_project, \
        mock.patch('cv2.imshow'), \
        mock.patch(
            'cv2.waitKey',
            side_effect=[ord('f'), ord('i'), ord('s'), 27]
        ) as mock_waitkey, \
        mock.patch(
            'cv2.rotate',
            return_value=np.zeros((500, 500, 3), dtype=np.uint8)
        ), \
        mock.patch('cv2.imwrite'), \
        mock.patch('src.reference.save_data'), \
        mock.patch(
            'src.reference.generate_object_points',
            return_value=sample_corners
        ), \
        mock.patch(
            'src.reference.generate_3d_axis',
            return_value=sample_3d_axis
        ), \
        mock.patch('src.reference.add_text_xyz'), \
        mock.patch('src.reference.draw_axis_3d'), \
        mock.patch(
            'src.reference.tvecs_to_xyz',
            return_value=sample_distances
        ), \
        mock.patch(
            'src.reference.rvecs_to_xyz_euler',
            return_value=sample_angles
        ), \
        mock.patch(
            'src.reference.generate_file_name',
            return_value='ref_jpg_of_chessboard_2024_10_31__10_10_10.jpg'
    ):

        # act
        reference_chessboard(
            sample_video_capture,
            sample_camera_matrix,
            sample_distortion_matrix,
            sample_settings,
        )

        # assert
        msg = 'cv2.cvtColor was not called'
        assert mock_cvtcolor.called, msg
        msg = 'cv2.findChessboardCorners was not called'
        assert mock_find_corners.called, msg
        msg = 'cv2.cornerSubPix was not called'
        assert mock_subpix.called, msg
        msg = 'cv2.drawChessboardCorners was not called'
        assert mock_drawtemplatecorners.called, msg
        msg = 'cv2.solvePnP was not called'
        assert mock_solvepnp.called, msg
        msg = 'cv2.projectPoints was not called'
        assert mock_project.called, msg
        assert mock_waitkey.call_count == 4
