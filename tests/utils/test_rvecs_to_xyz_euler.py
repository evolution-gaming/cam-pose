"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

import cv2 as cv
import numpy as np
import pytest

from src.utils import rvecs_to_xyz_euler


def test_rvecs_to_xyz_euler_no_singularity():
    # arrange (no singularity)
    rvec = np.array([0.1, 0.2, 0.3])

    # act
    euler_angles = rvecs_to_xyz_euler(rvec)

    # assert
    assert np.all(euler_angles >= 0)
    assert np.all(euler_angles < 360)


def test_rvecs_to_xyz_euler_invalid_input():
    # act and assert
    with pytest.raises(cv.error):
        rvecs_to_xyz_euler(np.array([1, 2]))
