"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

from typing import List, Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from .config import Settings
from .utils import (add_menu, generate_file_name, generate_object_points,
                    initialize_camera, save_data, setup_display_window)


def calculate_reprojection_errors(
    object_points: list,
    rotation_vectors: tuple,
    translation_vectors: tuple,
    camera_matrix: np.ndarray,
    distortion_coefficients: np.ndarray,
    image_points: list,
) -> List[float]:
    """
    Calculate the reprojection error between 3D object points and 2D image
    points.

    This function projects 3D object points onto the image plane using the
    camera's intrinsic and extrinsic parameters (rotation, translation, and
    distortion). It computes the L2 norm of the difference between the
    projected and actual image points.

    Args:
        object_points (list): List of 3D object points.
        rotation_vectors (tuple): List of rotation vectors for each camera
        pose.
        translation_vectors (tuple): List of translation vectors for each
        camera pose.
        camera_matrix (np.ndarray): Camera intrinsic matrix.
        distortion_coefficients (np.ndarray): Camera distortion coefficients.
        image_points (list): List of 2D image points corresponding to the
        object points.

    Returns:
        List[float]: List of reprojection errors for each set of object and
        image points.
    """
    # ensure that all input lists have the same length
    if not (len(object_points) == len(rotation_vectors) ==
            len(translation_vectors) == len(image_points)):
        raise ValueError('All input lists must have the same length.')

    errors = []
    for obj_pts, rvec, tvec, img_pts in zip(
        object_points, rotation_vectors, translation_vectors, image_points
    ):
        # transform the object points to image points
        projected_img_pts, _ = cv.projectPoints(
            obj_pts, rvec, tvec, camera_matrix, distortion_coefficients
        )

        # reshape projected points to match the shape and type of image_points
        projected_img_pts = projected_img_pts.reshape(-1, 2).astype(np.float32)
        img_pts = img_pts.reshape(-1, 2).astype(np.float32)

        # calculate absolute norm between transformation and
        # the corner finding algorithm
        error = cv.norm(
            img_pts, projected_img_pts, cv.NORM_L2
        ) / len(projected_img_pts)
        errors.append(error)
    return errors


def visualize_calibration_data(
    image_points: list, image_width: float, image_height: float, errors: list,
) -> None:
    """
    Visualize camera calibration data and reprojection errors.

    This function generates two plots:
    1. A scatter plot showing the coverage of the calibration data on the
    image.
    2. A bar chart displaying the reprojection error for each image, with an
    outlier limit.

    Args:
        image_points (list): List of 2D image points used in the calibration
        process.
        image_width (float): Width of the image in pixels.
        image_height (float): Height of the image in pixels.
        errors (list): List of reprojection errors for each calibration image.

    Returns:
        None: Displays the visualization plots.
    """
    if not errors:
        print('No re-projection errors to visualize\n')
        return

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(16, 9), gridspec_kw={'height_ratios': [2, 1]},
    )

    # calibration data coverage
    xy_total, xy_covered = image_width * image_height, set()
    for _, img_point in enumerate(image_points):
        img_point = img_point.reshape(-1, 2)
        x, y = img_point[:, 0], img_point[:, 1]
        xy_covered.update((round(x, 0), round(y, 0)) for x, y in zip(x, y))
        ax1.scatter(x, y, color='#dda15e', marker='x', s=25)
    xy_covered_len = len(xy_covered)
    xy_covered_perc = round((xy_covered_len/(xy_total)) * 100, 3)
    box_text = f'Covered coordinates: {xy_covered_len}, or {xy_covered_perc} %'
    ax1.set_xlim(0, image_width)
    ax1.set_ylim(0, image_height)
    ax1.set_xlabel('Image width, px')
    ax1.set_ylabel('Image height, px')
    # flip the Y-axis to match an intuitive camera movement visualization
    ax1.invert_yaxis()
    ax1.text(
        0.5,
        -0.15,
        box_text,
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )
    ax1.grid(True)
    ax1.set_title('Calibration data coverage')

    # re-projection error(s)
    image_numbers = np.arange(1, len(errors) + 1)
    ax2.bar(
        image_numbers,
        errors,
        color='#457b9d',
        alpha=0.7,
        label='Re-projection error',
    )
    mean_error, std_error = np.mean(errors), np.std(errors)
    outlier_lim = mean_error + 2 * std_error
    ax2.axhline(
        y=outlier_lim, color='#e63946', linestyle='--', label='Outlier limit',
    )
    ax2.set_xticks(image_numbers[0::10])
    ax2.set_ylim(0, max(errors) + 0.5)
    ax2.set_xlabel('Image number')
    ax2.set_ylabel('Re-projection error, px')
    ax2.text(
        0.5,
        -0.3,
        f'Mean error: {round(mean_error, 3)} px',
        transform=ax2.transAxes,
        fontsize=12,
        verticalalignment='top',
        horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
    )
    ax2.grid(True)
    ax2.set_title('Re-projection error for each image')
    ax2.legend()

    fig.tight_layout(pad=3.0)
    plt.show()


def delete_points(object_points: list, image_points: list) -> bool:
    """
    Delete a set of object and image points by their index.

    Prompts the user to enter the index of the points to delete. If the index
    is valid,
    the corresponding object and image points are removed from the lists.

    Args:
        object_points (list): List of 3D object points.
        image_points (list): List of 2D image points corresponding to the
        object points.

    Returns:
        bool: True if points were deleted, False if the index was invalid or
        input was incorrect.
    """
    is_deleted = False
    try:
        img_idx = int(input('Enter image index to delete: ')) - 1
        if img_idx < 0 or img_idx >= len(object_points):
            print(f'Index {img_idx + 1} is out of range\n')
            return is_deleted

        del object_points[img_idx]
        del image_points[img_idx]
        is_deleted = True
        print(
            f'Object and image points with index {img_idx + 1} '
            'have been deleted\n'
        )
        return is_deleted

    except ValueError:
        print('Invalid index. Please enter a valid integer\n')
        return is_deleted


def calibrate(
    object_points: list, image_points: list, image_size: list,
) -> tuple:
    """
    Calibrate the camera using 3D object points and 2D image points.

    This function performs camera calibration by computing the camera matrix,
    distortion coefficients, rotation vectors, and translation vectors. It uses
    the OpenCV function `cv.calibrateCamera` to obtain these parameters.

    Args:
        object_points (list): List of 3D object points used for calibration.
        image_points (list): List of 2D image points corresponding to the
        object points.
        image_size (list): Size of the calibration images in pixels (width,
        height).

    Returns:
        tuple: A tuple containing:
            - ret (bool): Success flag indicating if calibration was
            successful.
            - cam_mtx (np.ndarray): Camera matrix.
            - dist_coeffs (np.ndarray): Distortion coefficients.
            - rvecs (tuple): Rotation vectors for each camera pose.
            - tvecs (tuple): Translation vectors for each camera pose.
    """
    ret = False
    cam_mtx, dist_coeffs, rvecs, tvecs = np.array([]), np.array([]), (), ()
    if object_points and image_points:
        print('Calibrating the camera...\n')
        ret, cam_mtx, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(
            objectPoints=object_points,
            imagePoints=image_points,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None,
        )
        print('Camera has been calibrated\n')
    else:
        print('No points to perform camera calibration\n')
    return ret, cam_mtx, dist_coeffs, rvecs, tvecs


def calibration_chessboard(
    video_capture: cv.VideoCapture,
    object_points: list,
    image_points: list,
    img_size: Tuple[int, int],
    configuration: Settings,
) -> tuple:
    """
    Perform camera calibration using a chessboard pattern for data collection.

    This function captures video frames, detects the chessboard corners, and
    collects object and image points to be used for camera calibration. It
    also displays the chessboard detection process and allows the user to exit
    by pressing the 'ESC' key. Once enough data is collected, it performs
    the camera calibration using the collected points.

    Args:
        video_capture (cv.VideoCapture): OpenCV VideoCapture object for
        capturing video frames.
        object_points (list): List to store the 3D object points (chessboard
        pattern) for calibration.
        image_points (list): List to store the 2D image points (detected
        chessboard corners) for calibration.
        img_size (Tuple[int, int]): The size of the calibration images (width,
        height).
        configuration (Settings): Configuration object containing chessboard
        parameters and termination criteria.

    Returns:
        tuple: A tuple containing:
            - ret (bool): Success flag indicating if the calibration was
            successful.
            - cam_mtx (np.ndarray): Camera matrix.
            - dist_coeffs (np.ndarray): Distortion coefficients.
            - rvecs (tuple): Rotation vectors for each camera pose.
            - tvecs (tuple): Translation vectors for each camera pose.
            - object_points (list): Updated list of 3D object points.
            - image_points (list): Updated list of 2D image points.
    """
    ret = False
    cam_mtx, dist_coeffs, rvecs, tvecs = np.array([]), np.array([]), (), ()

    objp = generate_object_points(configuration)

    menu_description = {'ESC': 'Escape'}

    while True:
        _, img = video_capture.read()

        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        board_found, corners = cv.findChessboardCorners(
            img_gray,
            (configuration.CB_ROWS, configuration.CB_COLUMNS),
            None,
        )
        if board_found:
            object_points.append(objp)

            corn_det = cv.cornerSubPix(
                image=img_gray,
                corners=corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=configuration.TERM_CRIT,
            )
            image_points.append(corn_det)

            cv.drawChessboardCorners(
                img,
                (configuration.CB_ROWS, configuration.CB_COLUMNS),
                corn_det,
                board_found,
            )

        add_menu(img, menu_description, configuration.WHITE)

        cv.imshow(configuration.CB_PATT, img)

        if cv.waitKey(200) == 27:
            # <escape> key to exit
            print('Leaving data collection process...\n')
            break

    ret, cam_mtx, dist_coeffs, rvecs, tvecs = calibrate(object_points,
                                                        image_points, img_size)

    return ret, cam_mtx, dist_coeffs, rvecs, tvecs, object_points, image_points


def calibration_menu(camera_index: int, configuration: Settings) -> None:
    """
    Displays a menu for camera calibration, allowing the user to calibrate the
    camera, delete calibration points, save calibration data, or visualize
    calibration results.

    Args:
        camera_index (int): Index of the camera to be used for calibration.
        configuration (Settings): Configuration settings for the camera and
        calibration process.

    Key Press Options:
        'C' : Start the calibration process using chessboard images.
        'D' : Delete previously captured calibration points.
        'S' : Save the calibration data.
        'V' : Visualize the calibration data and reprojection errors.
        ESC : Exit the calibration menu.

    Returns:
        None
    """
    img_size = None
    obj_points, img_points = [], []
    ret = False
    cam_mtx, dist_coeffs, rvecs, tvecs = np.array([]), np.array([]), (), ()

    menu_description = {
        'C': 'Calibration',
        'D': 'Delete points',
        'S': 'Save Calibration',
        'V': 'Visualize Data',
        'ESC': 'Escape',
    }

    cap = initialize_camera(camera_index, configuration)

    if cap:
        setup_display_window(configuration.CB_PATT, configuration.WINDOW_WIDTH,
                             configuration.WINDOW_HEIGHT)

        frame_width_actual = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        frame_height_actual = cap.get(cv.CAP_PROP_FRAME_HEIGHT)

        while cap.isOpened():
            _, img = cap.read()
            add_menu(img, menu_description, configuration.WHITE)
            cv.imshow(configuration.CB_PATT, img)

            if not img_size:
                img_size = img.shape[1::-1]

            k = cv.waitKey(5)
            if k == 27:
                # <escape> key to exit
                break
            elif k == ord('c'):
                (ret,
                 cam_mtx, dist_coeffs,
                 rvecs, tvecs, object_points,
                 image_points) = calibration_chessboard(cap, obj_points,
                                                        img_points, img_size,
                                                        configuration)
            elif k == ord('d'):
                is_deleted = delete_points(obj_points, img_points)
                if is_deleted:
                    (ret, cam_mtx,
                     dist_coeffs, rvecs,
                     tvecs) = calibrate(obj_points, img_points, img_size)
            elif k == ord('s'):
                file_name = generate_file_name(
                    'cal', 'pckl', configuration.CB_PATT,
                )
                save_data(file_name, cam_mtx, dist_coeffs, rvecs, tvecs)
            elif k == ord('v'):
                reprojection_errors = calculate_reprojection_errors(
                    obj_points, rvecs, tvecs, cam_mtx, dist_coeffs, img_points,
                )
                visualize_calibration_data(
                    img_points,
                    frame_width_actual,
                    frame_height_actual,
                    reprojection_errors,
                )

        cap.release()
        cv.destroyAllWindows()
        cv.waitKey(1)
