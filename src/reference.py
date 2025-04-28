"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

import cv2 as cv
import numpy as np

from .config import Settings
from .utils import (add_menu, add_text_xyz, chessboard_inverted, draw_axis_3d,
                    generate_3d_axis, generate_file_name,
                    generate_object_points, initialize_camera,
                    rvecs_to_xyz_euler, save_data, setup_display_window,
                    tvecs_to_xyz)


def reference_chessboard(
    video_capture: cv.VideoCapture,
    camera_matrix: np.ndarray,
    distortion_matrix: np.ndarray,
    configuration: Settings,
) -> None:
    """
    Captures and processes chessboard images to establish a reference for
    camera calibration. Allows the user to flip the image, save the current
    image, or save the reference data.

    Args:
        video_capture (cv.VideoCapture): The video capture object for reading
        frames from the camera.
        camera_matrix (np.ndarray): The camera matrix for undistortion and
        perspective calculations.
        distortion_matrix (np.ndarray): The distortion coefficients for lens
        distortion correction.
        configuration (Settings): Configuration settings such as chessboard
        pattern size and calibration criteria.

    Key Press Options:
        'F' : Flip the image vertically.
        'I' : Save the current image as a reference.
        'S' : Save the reference data (corners, distance, and angles).
        ESC : Exit the reference collection process.

    Returns:
        None
    """
    img_flipped = False

    dist, angl_euler = np.full(3, np.nan), np.full(3, np.nan)

    objp = generate_object_points(configuration)
    axis = generate_3d_axis(configuration.SQUARE_SIZE)

    menu_description = {
        'F': 'Flip Image',
        'I': 'Save Image',
        'S': 'Save Reference',
        'ESC': 'Escape',
    }

    while True:
        _, img = video_capture.read()
        k = cv.waitKey(5)

        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        board_found, corners = cv.findChessboardCorners(
            img_gray,
            (configuration.CB_ROWS, configuration.CB_COLUMNS),
            None,
        )

        if board_found:
            corn_det = cv.cornerSubPix(
                image=img_gray,
                corners=corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=configuration.TERM_CRIT,
            )

            cb_inv = chessboard_inverted(corn_det, configuration)
            if cb_inv:
                print('The template or camera may be inverted\n')

            cv.drawChessboardCorners(
                img,
                (configuration.CB_ROWS, configuration.CB_COLUMNS),
                corn_det,
                board_found,
            )

            _, rvecs, tvecs = cv.solvePnP(
                objp, corn_det, camera_matrix, distortion_matrix,
            )

            dist = tvecs_to_xyz(tvecs)
            angl_euler = rvecs_to_xyz_euler(rvecs)

            img_points, _ = cv.projectPoints(
                axis, rvecs, tvecs, camera_matrix, distortion_matrix,
            )

            draw_axis_3d(img, corn_det[0], img_points, configuration)

        if img_flipped:
            img = cv.rotate(img, cv.ROTATE_180)

        add_text_xyz(img, dist, angl_euler, configuration, 'Actual', 0)

        add_menu(img, menu_description, configuration.WHITE)

        cv.imshow(configuration.CB_PATT, img)

        if k == 27:
            # <escape> key to exit
            print('Leaving data collection process...\n')
            break
        elif k == ord('f'):
            img_flipped = not img_flipped
        elif k == ord('i'):
            img_name = generate_file_name('ref', 'jpg', configuration.CB_PATT)
            cv.imwrite(img_name, img)
        elif k == ord('s'):
            file_name = generate_file_name(
                'ref', 'pckl', configuration.CB_PATT,
            )
            save_data(file_name, corners, dist, angl_euler)


def reference_menu(
    camera_index: int,
    camera_matrix: np.ndarray,
    distortion_matrix: np.ndarray,
    configuration: Settings,
) -> None:
    """
    Displays a menu to initiate the reference chessboard data collection
    process. Allows the user to start capturing reference data for camera
    calibration.

    Args:
        camera_index (int): Index of the camera to be used for capturing
        images.
        camera_matrix (np.ndarray): The camera matrix for camera calibration.
        distortion_matrix (np.ndarray): The distortion coefficients for lens
        distortion correction.
        configuration (Settings): Configuration settings such as window size
        and chessboard pattern.

    Key Press Options:
        'R' : Start the reference data collection (chessboard capture).
        ESC : Exit the menu and terminate the process.

    Returns:
        None
    """
    menu_description = {'R': 'Reference', 'ESC': 'Escape'}

    cap = initialize_camera(camera_index, configuration)

    if cap:
        setup_display_window(configuration.CB_PATT, configuration.WINDOW_WIDTH,
                             configuration.WINDOW_HEIGHT)

        while cap.isOpened():
            _, img = cap.read()
            add_menu(img, menu_description, configuration.WHITE)
            cv.imshow(configuration.CB_PATT, img)

            k = cv.waitKey(5)
            if k == 27:
                # <escape> key to exit
                break
            elif k == ord('r'):
                reference_chessboard(
                    cap, camera_matrix, distortion_matrix, configuration,
                )

        cap.release()
        cv.destroyAllWindows()
        cv.waitKey(1)
