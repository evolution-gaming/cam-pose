"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

import logging
import pickle
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2 as cv
import numpy as np

from .config import Settings, config


def setup_logging() -> logging.RootLogger:
    """
    Configures and returns a logger for the application.

    Sets up a stream handler to output log messages to the console with
    the specified log format and log level from the configuration.

    Returns:
        logging.RootLogger: The configured root logger instance.
    """
    formatter = logging.Formatter(config.LOG_FORMAT)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(config.LOG_LEVEL)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def detect_camera_indexes() -> list:
    """
    Detects and returns a list of indexes for connected cameras.

    Attempts to open camera devices with indexes from 0 to 8 and appends
    the valid indexes to a list.

    Returns:
        list: A list of indexes of connected cameras.
    """
    camera_indexes = []
    for idx in range(9):
        cap = cv.VideoCapture(idx)
        if cap.isOpened():
            camera_indexes.append(idx)
            cap.release()
    return camera_indexes


def select_camera_index() -> Optional[int]:
    """
    Prompts the user to select a valid camera index and returns it.

    Continuously asks for input until a valid index (integer) is provided.

    Returns:
        Optional[int]: The selected camera index.
    """
    cam_idx = None
    while True:
        cam_idx = input('Select camera index: ')
        if cam_idx.isdigit():
            cam_idx = int(cam_idx)
            print(f'\nSelected camera index {cam_idx}\n')
            break
        else:
            print('\nInvalid input\n')
    return cam_idx


def setup_display_window(pattern: str, width: int, height: int) -> None:
    """
    Configures the display window with specified size and pattern name.

    Args:
        pattern (str): The name of the window pattern.
        width (int): The desired width of the window.
        height (int): The desired height of the window.

    Returns:
        None
    """
    cv.namedWindow(pattern, cv.WINDOW_NORMAL)
    cv.resizeWindow(pattern, width, height)


def initialize_camera(
    camera_index: int, configuration: Settings,
) -> Optional[cv.VideoCapture]:
    """
    Initializes the camera with the specified settings and returns the video
    capture object.

    Args:
        camera_index (int): The index of the camera to be initialized.
        configuration (Settings): The configuration object containing camera
        settings.

    Returns:
        Optional[cv.VideoCapture]: The video capture object if successful,
        otherwise None.
    """
    # prepare capture settings
    if platform.system() == 'Windows':
        capture = cv.VideoCapture(camera_index, cv.CAP_DSHOW)
    else:
        capture = cv.VideoCapture(camera_index)
    if not capture.isOpened():
        print('Cannot initialize the camera\n')
        return None

    # set desired frame width and height
    capture.set(cv.CAP_PROP_FRAME_WIDTH, configuration.FRAME_WIDTH)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, configuration.FRAME_HEIGHT)
    return capture


def generate_3d_axis(axis_size: float) -> np.ndarray:
    """
    Generates a 3D axis with the specified size.

    Args:
        axis_size (float): The length of each axis.

    Returns:
        np.ndarray: A 3D axis represented as a numpy array of shape (3, 3).
    """
    return np.float32([
        [axis_size, 0, 0], [0, axis_size, 0], [0, 0, -axis_size]
    ]).reshape(-1, 3)


def generate_object_points(configuration: Settings) -> np.ndarray:
    """
    Generates 3D object points for a chessboard pattern.

    Args:
        configuration (Settings): The configuration object containing
        chessboard dimensions and square size.

    Returns:
        np.ndarray: An array of 3D object points shape:
        (CB_ROWS * CB_COLUMNS, 3)), with Z=0.
    """
    object_points = np.zeros(
        (configuration.CB_ROWS * configuration.CB_COLUMNS, 3), np.float32
    )
    grid = np.mgrid[0:configuration.CB_ROWS, 0:configuration.CB_COLUMNS]
    object_points[:, :2] = grid.T.reshape(-1, 2) * configuration.SQUARE_SIZE
    return object_points


def generate_file_name(title: str, extension: str, pattern: str) -> str:
    """
    Generates a formatted file name with the current timestamp.

    Args:
        title (str): The base name or title for the file.
        extension (str): The file extension.
        pattern (str): The pattern or identifier related to the file.

    Returns:
        str: A formatted file name, including the timestamp.
    """
    dt_now = datetime.now(timezone.utc).strftime('%Y_%m_%d__%H_%M_%S')
    return f'{title}_{extension}_of_{pattern}_{dt_now}.{extension}'


def save_data(file_name: str, *args: Union[np.ndarray, list]) -> None:
    """
    Saves data to a specified file using pickle.

    Args:
        file_name (str): The name of the file to save the data.
        *args (Union[np.ndarray, list]): The data to be saved
        (could be np.ndarray or list).

    Returns:
        None

    Raises:
        IOError: If an I/O error occurs during file operations.
        Exception: For any other errors during the save process.
    """
    try:
        with open(file_name, 'wb') as f:
            pickle.dump(args, f)
        print(f'Data has been saved as {file_name}\n')
    except IOError as io_e:
        print(f'IO error occurred while saving {file_name} data file: {io_e}')
    except Exception as e:
        print(f'An error occurred while saving {file_name} data file: {e}\n')


def load_data(file_name: str) -> tuple:
    """
    Loads data from a pickle file.

    Args:
        file_name (str): The name of the file to load the data from.

    Returns:
        tuple: The data loaded from the file, or an empty tuple if an error
        occurs.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        pickle.UnpicklingError: If the file is not a valid pickle file.
        Exception: For any other errors during the load process.
    """
    data: tuple = tuple()
    try:
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(f'File {file_name} not found\n')
    except pickle.UnpicklingError:
        print(f'The file {file_name} may not be a valid pickle file\n')
    except Exception as e:
        print(f'An error occurred while loading data: {e}\n')
    return data


def display_saved_data(
    directory: Path = Path.cwd(), file_format: str = 'pckl',
) -> List[str]:
    """
    Displays and returns a list of saved data files with the specified format
    in a directory.

    Args:
        directory (Path): The directory to search for saved data files (default
        is current working directory).
        file_format (str): The file format to filter (default is 'pckl').

    Returns:
        List[str]: A list of file names matching the specified format in the
        directory.
    """
    files = []
    try:
        for f in directory.iterdir():
            if f.is_file() and f.suffix == f'.{file_format}':
                files.append(f.name)

        if not files:
            print('No saved data found in the specified directory\n')
            return files

        print('Available saved data:')
        for idx, file in enumerate(files, 1):
            print(f'{idx} - {file}')
        print()

        return files

    except OSError as e:
        print(f'Error accessing directory {directory}: {e}')
        return files


def select_saved_data(files: list) -> Optional[str]:
    """
    Prompts the user to select a saved data file from a list by entering its
    number.

    Args:
        files (list): A list of available saved data file names.

    Returns:
        Optional[str]: The selected file name if a valid selection is made, or
        None if invalid input is provided.
    """
    if not files:
        return None
    selected_idx = input('Enter the number of the saved data: ')
    try:
        selected_file = files[int(selected_idx) - 1]
        if selected_file:
            print(f'\nSaved data {selected_file} selected\n')
            return selected_file
        else:
            print('\nNo valid selection made\n')
            return None
    except ValueError:
        print('\nInvalid input\n')
        return None
    except IndexError:
        print(
            f'\nSaved data with number {selected_idx} does not exist\n'
        )
        return None


def tvecs_to_xyz(translation_vectors: np.ndarray) -> np.ndarray:
    """
    Converts translation vectors to a 1D array of shape (3,).

    Args:
        translation_vectors (np.ndarray): A translation vector with shape (3,)
        or (3, 1).

    Returns:
        np.ndarray: A flattened 1D array with the translation vector.
    """
    if translation_vectors.shape == (3, 1):
        return translation_vectors.flatten()
    elif translation_vectors.shape == (3,):
        return translation_vectors
    else:
        msg = 'Input translation_vectors must be of shape (3,) or (3, 1)'
        raise ValueError(msg)


def rvecs_to_xyz_euler(rotation_vectors: np.ndarray) -> np.ndarray:
    """
    Computes Euler angles (roll, pitch, yaw) from rotation vectors.

    Args:
        rotation_vectors (np.ndarray): A rotation vector (3x1) or (3,)
        representing a rotation.

    Returns:
        np.ndarray: A 1D array of Euler angles (roll, pitch, yaw) in degrees,
        adjusted to be within [0, 360).
    """
    rotation_vectors, _ = cv.Rodrigues(rotation_vectors)
    sy = np.sqrt(rotation_vectors[0, 0] ** 2 + rotation_vectors[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(rotation_vectors[2, 1], rotation_vectors[2, 2])
        y = np.arctan2(-rotation_vectors[2, 0], sy)
        z = np.arctan2(rotation_vectors[1, 0], rotation_vectors[0, 0])
    else:
        x = np.arctan2(-rotation_vectors[1, 2], rotation_vectors[1, 1])
        y = np.arctan2(-rotation_vectors[2, 0], sy)
        z = 0
    euler_angles_degrees = np.degrees(np.array([x, y, z]))
    return np.where(
        euler_angles_degrees < 0,
        euler_angles_degrees + 360,
        euler_angles_degrees
    )


def chessboard_inverted(corners: np.ndarray, configuration: Settings) -> bool:
    """
    Checks if the chessboard pattern is inverted based on corner positions.

    Args:
        corners (np.ndarray): The detected corners of the chessboard pattern.
        configuration (Settings): Configuration object containing chessboard
        dimensions.

    Returns:
        bool: True if the chessboard is inverted, False otherwise.
    """
    inverted = False
    pattern_size = (configuration.CB_ROWS, configuration.CB_COLUMNS)

    # calculate vector between two points horizontally and vertically
    top_left = corners[0][0]
    top_right = corners[pattern_size[0] - 1][0]
    bottom_left = corners[pattern_size[0] * (pattern_size[1] - 1)][0]

    # horizontal vector: from the top-left to the top-right
    # should point to the right
    horizontal_vector = top_right - top_left

    # vertical vector: from the top-left to the bottom-left
    # should point downward
    vertical_vector = bottom_left - top_left

    # ensure both vectors are positive in X and Y respectively
    if horizontal_vector[0] <= 0 and vertical_vector[1] <= 0:
        inverted = True
    return inverted


def is_within_tolerance(
    distances_difference: np.ndarray,
    angles_difference: np.ndarray,
    tolerance: float
) -> bool:
    """
    Checks if all values in both distance and angle difference arrays are
    within the specified tolerance.

    Args:
        distances_difference (np.ndarray): Array of differences in distances.
        angles_difference (np.ndarray): Array of differences in angles.
        tolerance (float): The tolerance threshold.

    Returns:
        bool: True if all differences are within the tolerance, False
        otherwise.
    """
    return np.all(np.abs(distances_difference) < tolerance) and \
        np.all(np.abs(angles_difference) < tolerance)


def draw_axis_3d(
    image: np.ndarray,
    corners: np.ndarray,
    image_points: np.ndarray,
    configuration: Settings,
) -> np.ndarray:
    """
    Draws a 3D axis on the given image.

    Args:
        image (np.ndarray): The image on which the 3D axis will be drawn.
        corners (np.ndarray): The detected chessboard corners.
        image_points (np.ndarray): The projected 3D axis points on the image.
        configuration (Settings): Configuration object containing axis colors.

    Returns:
        np.ndarray: The image with the drawn 3D axis.
    """
    thickness: int = 5
    corner = tuple(corners.ravel().astype(int))
    axis_colors = [configuration.BLUE, configuration.GREEN, configuration.RED]
    if image_points.shape[0] >= 3:
        for i, color in enumerate(axis_colors):
            end_point = tuple(image_points[i].ravel().astype(int))
            cv.line(image, corner, end_point, color, thickness)
    return image


def draw_corners(
    image: np.ndarray, corners: np.ndarray, color: tuple,
) -> None:
    """
    Draws circles on the image at the given corner points.

    Args:
        image (np.ndarray): The image on which corners will be drawn.
        corners (np.ndarray): The detected corner points to be marked.
        color (tuple): The color of the circles to draw.

    Returns:
        None
    """
    for corner in corners:
        cv.circle(image, tuple(map(int, corner.ravel())), 5, color, -1)


def draw_principal_point(
    image: np.ndarray,
    camera_matrix: np.ndarray,
    label: str = 'principal point',
    color: tuple = (0, 0, 255),
) -> None:
    """
    Draws the principal point on the image using the camera matrix.

    Args:
        image (np.ndarray): The image on which the principal point will be
        drawn.
        camera_matrix (np.ndarray): The camera matrix containing the principal
        point coordinates.
        label (str): The label to display next to the principal point (default
        is 'principal point').
        color (tuple): The color of the circle and label (default is red).

    Returns:
        None
    """
    if camera_matrix.shape != (3, 3):
        raise ValueError('Expected camera matrix to be a 3x3 array.')

    x, y = int(camera_matrix[0, 2]), int(camera_matrix[1, 2])
    principal_point = (x, y)

    cv.circle(image, principal_point, radius=3, color=color, thickness=-1)
    cv.putText(
        image,
        label,
        (principal_point[0] + 10, principal_point[1] - 10),
        cv.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2,
    )


def add_text_xyz(
    image: np.ndarray,
    distances: np.ndarray,
    angles: np.ndarray,
    configuration: Settings,
    title: str,
    x_shift_step: int,
    x_shift_start: int = 30,
) -> None:
    """
    Adds X, Y, Z distance and angle information to the image.

    Args:
        image (np.ndarray): The image to display the text on.
        distances (np.ndarray): The X, Y, Z distance values.
        angles (np.ndarray): The roll, pitch, and yaw angle values.
        configuration (Settings): The configuration containing units and color
        settings.
        title (str): The title to display at the top.
        x_shift_step (int): The horizontal offset for the text.
        x_shift_start (int): The starting horizontal position for the text
        (default is 30).

    Returns:
        None
    """
    labels, angle_labels = ['x', 'y', 'z'], ['r', 'p', 'y']
    colors = [configuration.BLUE, configuration.GREEN, configuration.RED]

    cv.putText(
        image,
        title,
        (x_shift_start + x_shift_step, 30),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        configuration.WHITE,
        2,
    )

    for i, (label, angle_label, color) in enumerate(zip(labels, angle_labels, colors)):  # noqa
        dist_txt = (
            f'{label}='
            f'{round(distances[i], 2) if not np.isnan(distances[i]) else "NA"}'
            f' {configuration.DIST_UNIT}'
        )
        angl_txt = (
            f'{angle_label}='
            f'{round(angles[i], 2) if not np.isnan(angles[i]) else "NA"}'
            f' {configuration.ANGLE_UNIT}'
        )

        cv.putText(
            image,
            f'{dist_txt}, {angl_txt}',
            (x_shift_start + x_shift_step, 60 + i * 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )


def add_menu(
    image: np.ndarray, menu: Dict[str, str], color: Tuple[int, ...],
) -> None:
    """
    Adds a menu overlay to the image.

    Args:
        image (np.ndarray): The image to display the menu on.
        menu (Dict[str, str]): A dictionary containing menu options and their
        descriptions.
        color (Tuple[int, ...]): The color of the text to be displayed.

    Returns:
        None
    """
    line_height = 30
    for i, (k, v) in enumerate(menu.items()):
        cv.putText(
            image,
            f'[{k}] {v}',
            (image.shape[1] - 200,
             image.shape[0] - (len(menu) * line_height) + i * line_height),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            1,
        )
