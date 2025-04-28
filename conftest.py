"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

from unittest import mock

import cv2 as cv
import numpy as np
import pytest

from src.position import NavigationBall
from tests.sample_config import SampleSettings, sample_config


@pytest.fixture
def sample_settings() -> SampleSettings:
    return sample_config


@pytest.fixture
def sample_image() -> np.ndarray:
    return np.zeros((500, 500, 3), dtype=np.uint8)


@pytest.fixture
def sample_video_capture(sample_image) -> mock.MagicMock:
    mock_capture = mock.MagicMock(spec=cv.VideoCapture)
    mock_capture.read.return_value = (True, sample_image)
    return mock_capture


@pytest.fixture
def sample_3d_axis() -> np.ndarray:
    return np.zeros((4, 3), np.float32)


@pytest.fixture
def sample_image_points() -> np.ndarray:
    return np.array([[[60, 50]], [[50, 60]], [[50, 40]]], dtype=np.float32)


@pytest.fixture
def sample_corners() -> np.ndarray:
    return np.array([[50, 50]], dtype=np.float32)


@pytest.fixture
def sample_distances() -> np.ndarray:
    return np.array([1.23, 4.56, 7.89])


@pytest.fixture
def sample_angles() -> np.ndarray:
    return np.array([10.11, 20.22, 30.33])


@pytest.fixture
def sample_distances_nan() -> np.ndarray:
    return np.array([np.nan, 4.56, 7.89])


@pytest.fixture
def sample_angles_nan() -> np.ndarray:
    return np.array([10.11, np.nan, 30.33])


@pytest.fixture
def sample_nan_all() -> np.ndarray:
    return np.full(3, np.nan)


@pytest.fixture
def sample_camera_matrix() -> np.ndarray:
    return np.array([
        [1, 0, 250],
        [0, 1, 250],
        [0, 0, 1]
    ], dtype=np.float32)


@pytest.fixture
def sample_distortion_matrix() -> np.ndarray:
    return np.array([-0.5, 0.3, 0.01, -0.02, 0.1], dtype=np.float32)


@pytest.fixture
def sample_calibration_set() -> tuple:
    sample_obj_points = [[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]]
    sample_img_points = [[[10, 20]], [[30, 40]], [[50, 60]]]
    sample_img_size = (640, 480)
    return sample_obj_points, sample_img_points, sample_img_size


@pytest.fixture
def sample_vectors() -> tuple:
    sample_tvecs = (np.array(
        [[-130.12996216], [13.74905281], [530.38258116]],
        dtype=np.float32
    ),)
    sample_rvecs = (np.array(
        [[-0.00843164], [0.20176136], [2.99714999]], dtype=np.float32
    ),)
    return sample_tvecs, sample_rvecs


@pytest.fixture
def sample_data_points() -> tuple:
    sample_obj_p = [np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)]
    sample_img_p = [np.array([[0, 0], [1, 1]], dtype=np.float32)]
    return sample_obj_p, sample_img_p


@pytest.fixture
def nav_ball(sample_image, sample_settings) -> NavigationBall:
    return NavigationBall(
        image=sample_image,
        widget_size=sample_settings.NAV_BALL_WDGT_SIZE,
        ball_color=sample_settings.NAV_BALL_CLR,
        background_color=sample_settings.NAV_BALL_BCKGRND_CLR,
        tolerance=sample_settings.MEAS_TOL,
        distance_unit=sample_settings.DIST_UNIT,
        angle_unit=sample_settings.ANGLE_UNIT,
    )
