"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

import logging
from typing import Tuple

import cv2 as cv
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Configuration class for the camera calibration tool.

    This class inherits from `BaseSettings` and loads configuration values from
    the 'config.env' file. It stores various settings required for camera
    calibration, such as naming conventions, chessboard pattern parameters, and
    other configurations.
    """
    model_config = SettingsConfigDict(env_file='config.env')

    # namings
    CB_PATT: str = 'chessboard'

    # window
    WINDOW_WIDTH: int = 0
    WINDOW_HEIGHT: int = 0

    # camera
    FRAME_WIDTH: float = 0.0
    FRAME_HEIGHT: float = 0.0

    # object
    CB_ROWS: int = 0
    CB_COLUMNS: int = 0
    SQUARE_SIZE: float = 0.0

    # criteria
    TERM_CRIT: tuple = (
        cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001
    )

    # tolerances
    MEAS_TOL: float = 0.0

    # units
    DIST_UNIT: str = ''
    ANGLE_UNIT: str = ''

    # colors
    RED: tuple = (0, 0, 255)
    GREEN: tuple = (0, 255, 0)
    BLUE: tuple = (255, 0, 0)
    WHITE: tuple = (255, 255, 255)
    GREY: tuple = (150, 150, 150)
    BLACK: tuple = (0, 0, 0)

    # nav-ball
    NAV_BALL_WDGT_SIZE: int = 0
    NAV_BALL_CLR: Tuple[int, int, int] = 0, 0, 0
    NAV_BALL_BCKGRND_CLR: Tuple[int, int, int] = 0, 0, 0

    # logging
    LOG_FORMAT: str = (
        '%(asctime)s '
        '[%(levelname)s] '
        '{filename: %(filename)s, '
        'function: %(funcName)s(), '
        'message: %(message)s}'
    )
    LOG_LEVEL: int = logging.INFO


config = Settings()
