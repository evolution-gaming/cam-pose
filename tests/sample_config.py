"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

from typing import Tuple

import cv2 as cv


class SampleSettings:
    """
    Sample configuration class for the camera calibration tool unittests.
    """
    # namings
    CB_PATT: str = 'chessboard'

    # window
    WINDOW_WIDTH: int = 800
    WINDOW_HEIGHT: int = 600

    # camera
    FRAME_WIDTH: float = 640.0
    FRAME_HEIGHT: float = 480.0

    # object
    CB_ROWS: int = 5
    CB_COLUMNS: int = 7
    SQUARE_SIZE: float = 25.0

    # criteria
    TERM_CRIT: tuple = (
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001
    )

    # tolerances
    MEAS_TOL: float = 0.5

    # units
    DIST_UNIT: str = 'mm'
    ANGLE_UNIT: str = 'deg'

    # colors
    RED: tuple = (0, 0, 255)
    GREEN: tuple = (0, 255, 0)
    BLUE: tuple = (255, 0, 0)
    WHITE: tuple = (255, 255, 255)
    GREY: tuple = (150, 150, 150)
    BLACK: tuple = (0, 0, 0)

    # nav-ball
    NAV_BALL_WDGT_SIZE: int = 500
    NAV_BALL_CLR: Tuple[int, int, int] = 150, 150, 150
    NAV_BALL_BCKGRND_CLR: Tuple[int, int, int] = 0, 0, 0


sample_config = SampleSettings()
