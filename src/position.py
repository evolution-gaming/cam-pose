"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

from typing import Union

import cv2 as cv
import numpy as np

from .config import Settings
from .utils import (add_menu, add_text_xyz, chessboard_inverted, draw_axis_3d,
                    draw_corners, draw_principal_point, generate_3d_axis,
                    generate_file_name, generate_object_points,
                    initialize_camera, is_within_tolerance, rvecs_to_xyz_euler,
                    setup_display_window, tvecs_to_xyz)


def compute_angle_difference(
    angle_actual: Union[np.ndarray, float],
    angle_reference: Union[np.ndarray, float],
) -> Union[np.ndarray, float]:
    """
    Calculate the normalized difference between two angles in the range
    [-180, +180].

    This function:
    1. Computes the difference between the actual and reference angles.
    2. Normalizes the result to the range [-180, +180] using modulo arithmetic.

    Args:
        angle_actual (np.ndarray, float): The actual angle value.
        angle_reference (np.ndarray, float): The reference angle value.

    Returns:
        np.ndarray, float: The angle difference in the range [-180.0, +180.0].
    """
    return (angle_actual - angle_reference + 180) % 360 - 180


class NavigationBall:
    """
    Direction Finder is a class that visually assists with camera positioning.
    """
    _RGB: dict = {'X': (255, 0, 0), 'Y': (0, 255, 0), 'Z': (0, 0, 255)}

    def __init__(
        self,
        image: np.ndarray,
        widget_size: int,
        ball_color: tuple,
        background_color: tuple,
        tolerance: float,
        distance_unit: str,
        angle_unit: str,
    ) -> None:
        self.img = image
        self.widget_size = widget_size
        self.ball_color = ball_color
        self.background_color = background_color
        self.tolerance = tolerance
        self.dist_unit = distance_unit
        self.angl_unit = angle_unit

        self.thick = 2
        self.offset_step = 10
        self.offset = (self.offset_step,
                       image.shape[0] - widget_size - self.offset_step)
        self.center = (self.offset[0] + widget_size // 2,
                       self.offset[1] + widget_size // 2)
        self.radius = widget_size // 3

        self.dash_len = 10
        self.dash_gap_len = 10
        self.crcl_arr_len = 10
        self.max_arr_len = 90
        self.fixed_tip_length = 10

    def _draw_dashed_lines(self, start_point: tuple, end_point: tuple) -> None:
        """
        Draw dashed lines between two points on the image.

        This function calculates the distance between the start and end points,
        divides it into segments, and draws individual dashed segments between
        them.

        Args:
            start_point (tuple): The (x, y) coordinates of the line's start
            point.
            end_point (tuple): The (x, y) coordinates of the line's end point.

        Returns:
            None
        """
        dist = np.linalg.norm(np.array(end_point) - np.array(start_point))
        num_dashes = int(dist // (self.dash_len + self.dash_gap_len))
        for i in range(num_dashes):
            start_dash = (
                int(
                    start_point[0]
                    + (end_point[0] - start_point[0]) * (i / num_dashes)
                ),
                int(
                    start_point[1]
                    + (end_point[1] - start_point[1]) * (i / num_dashes)
                ),
            )
            end_dash = (
                int(
                    start_point[0]
                    + (end_point[0] - start_point[0])
                    * ((i + 0.5) / num_dashes)
                ),
                int(
                    start_point[1]
                    + (end_point[1] - start_point[1])
                    * ((i + 0.5) / num_dashes)
                ),
            )
            cv.line(
                self.img, start_dash, end_dash, self.ball_color, self.thick,
            )

    def _draw_template(self) -> None:
        """
        Draw a visual template consisting of:
        - A square background
        - A circle (Z axis)
        - Two ellipses (X and Y axes)
        - Dashed lines indicating the directions along X, Y, and Z axes
        - A center point

        This function helps visualize the reference template used in the
        calibration or visualization process.

        Args:
            None

        Returns:
            None
        """
        # background
        cv.rectangle(
            self.img,
            self.offset,
            (self.offset[0] + self.widget_size,
             self.offset[1] + self.widget_size),
            self.background_color,
            -1,
        )
        # circle Z
        cv.circle(
            self.img, self.center, self.radius, self.ball_color, self.thick,
        )
        # ellipses X and Y
        axes_X = (self.radius, int(self.radius * 0.3))
        axes_Y = (int(self.radius * 0.3), self.radius)
        cv.ellipse(self.img, self.center, axes_X, 0,
                   0, 360, self.ball_color, self.thick)
        cv.ellipse(self.img, self.center, axes_Y, 0,
                   0, 360, self.ball_color, self.thick)
        # dashes for X, Y and Z
        directions = [((
            self.center[0] - self.radius, self.center[1]
        ), (
            self.center[0] + self.radius, self.center[1]
        )), ((
            self.center[0], self.center[1] - self.radius
        ), (
            self.center[0], self.center[1] + self.radius
        )), ((
            self.center[0] - int(self.radius * 0.7),
            self.center[1] - int(self.radius * 0.7)
        ), (
            self.center[0] + int(self.radius * 0.7),
            self.center[1] + int(self.radius * 0.7)
        ))]
        for start, end in directions:
            self._draw_dashed_lines(start, end)
        # center point
        cv.circle(self.img, self.center, 5, self.ball_color, -1)

    def _highlight_angle(self, axis: str, angle_diff: float) -> None:
        """
        Draws the angle difference on the image for a specific axis (X, Y, Z).
        Displays the angle difference as text and visualizes it using an arc
        and an arrow indicating the direction of the angle change.

        This function helps visualize the deviation of an object's orientation
        relative to a reference axis.

        Args:
            axis (str): The axis to highlight ('X', 'Y', or 'Z').
            angle_diff (float): The calculated angle difference in degrees.

        Returns:
            None
        """
        cv.putText(self.img,
                   f'{axis.lower()}: {angle_diff:.2f} {self.angl_unit}',
                   (30,
                    30
                    + self.img.shape[0]
                    - self.widget_size
                    + {'X': 0, 'Y': 30, 'Z': 60}[axis]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, self._RGB[axis], 1)

        if np.isnan(angle_diff):
            return
        if abs(angle_diff) <= self.tolerance:
            return

        angle_deg = int(angle_diff / self.tolerance)
        start_angle = {'X': -90, 'Y': 180, 'Z': 0}[axis]
        end_angle = None
        if axis == 'Y':
            end_angle = start_angle + angle_deg
        else:
            end_angle = start_angle - angle_deg

        axes = {
            'X': (int(self.radius * 0.3), self.radius),
            'Y': (self.radius, int(self.radius * 0.3)),
            'Z': (self.radius, self.radius)
        }[axis]
        cv.ellipse(self.img, self.center, axes, 0,
                   start_angle, end_angle, self._RGB[axis], self.thick + 1)

        arrow_tip, arrow_end = self._calculate_arrow_position(axis, angle_diff)
        cv.arrowedLine(self.img, arrow_tip, arrow_end,
                       self._RGB[axis], self.thick, tipLength=1)

    def _calculate_arrow_position(self, axis: str, angle_diff: float) -> tuple:
        """
        Calculates the start and end positions for an arrow based on the angle
        difference and the specified axis ('X', 'Y', or 'Z').

        Args:
            axis (str): The axis for which the arrow position is calculated
            ('X', 'Y', 'Z').
            angle_diff (float): The angle difference used to determine the
            arrow's direction.

        Returns:
            tuple: A tuple containing the (x, y) coordinates of the arrow's tip
            and end positions.
        """
        if axis == 'X':
            if angle_diff < 0:
                arrow_tip = (
                    self.center[0] + self.crcl_arr_len,
                    self.center[1] - self.radius + self.crcl_arr_len // 2
                )
                arrow_end = (
                    arrow_tip[0] - self.crcl_arr_len,
                    int(arrow_tip[1] - self.crcl_arr_len * 0.5)
                )
            else:
                arrow_tip = (
                    self.center[0] - self.crcl_arr_len,
                    self.center[1] - self.radius + self.crcl_arr_len // 2
                )
                arrow_end = (
                    arrow_tip[0] + self.crcl_arr_len,
                    int(arrow_tip[1] - self.crcl_arr_len * 0.5)
                )
        elif axis == 'Y':
            crcl_arr_len = self.crcl_arr_len // 2
            if angle_diff > 0:
                arrow_tip = (
                    self.center[0] - self.radius + crcl_arr_len,
                    self.center[1] - crcl_arr_len - (crcl_arr_len + 2)
                )
                arrow_end = (
                    arrow_tip[0] - crcl_arr_len,
                    int(arrow_tip[1] + crcl_arr_len * 2.4)
                )
            else:
                arrow_tip = (
                    self.center[0] - self.radius + crcl_arr_len,
                    int(self.center[1] + crcl_arr_len * 2.4)
                )
                arrow_end = (
                    arrow_tip[0] - crcl_arr_len,
                    arrow_tip[1] - crcl_arr_len - (crcl_arr_len + 2)
                )
        elif axis == 'Z':
            if angle_diff > 0:
                arrow_tip = (
                    self.center[0] + self.radius,
                    self.center[1] - self.crcl_arr_len
                )
                arrow_end = (arrow_tip[0], arrow_tip[1] + self.crcl_arr_len)
            else:
                arrow_tip = (
                    self.center[0] + self.radius,
                    self.center[1] + self.crcl_arr_len
                )
                arrow_end = (arrow_tip[0], arrow_tip[1] - self.crcl_arr_len)
        return arrow_tip, arrow_end

    def _highlight_distance(self, axis: str, distance_diff: float) -> None:
        """
        Highlights the distance difference along a specified axis by drawing an
        arrow on the image, indicating the distance and labeling it.

        Args:
            axis (str): The axis for which the distance difference is
            highlighted ('X', 'Y', 'Z').
            distance_diff (float): The calculated distance difference to be
            visualized.

        Returns:
            None: Modifies the image in place by drawing text and an arrow.
        """
        cv.putText(self.img,
                   f'{axis.lower()}: {distance_diff:.2f} {self.dist_unit}',
                   (self.center[0] + 100,
                    30
                    + self.img.shape[0]
                    - self.widget_size
                    + {'X': 0, 'Y': 30, 'Z': 60}[axis]),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, self._RGB[axis], 1)

        if np.isnan(distance_diff):
            return
        if abs(distance_diff) <= self.tolerance:
            return

        length = int(
            min(abs(distance_diff / self.max_arr_len), 1) * self.radius
        )
        if axis == 'Z':
            length = int(length * 0.7)

        end_point = self._calculate_distance_point(axis, distance_diff, length)
        tip_length = self._normalize_tip_length(end_point)
        cv.arrowedLine(self.img, end_point, self.center,
                       self._RGB[axis], self.thick, tipLength=tip_length)

    def _calculate_distance_point(
        self, axis: str, distance_diff: float, length: int,
    ) -> tuple:
        """
        Calculates the endpoint coordinates for visualizing the distance
        difference along a specified axis (X, Y, or Z).

        Args:
            axis (str): The axis along which the distance is calculated ('X',
            'Y', 'Z').
            distance_diff (float): The calculated distance difference.
            length (int): The length of the arrow indicating the distance.

        Returns:
            tuple: The (x, y) coordinates of the calculated endpoint based on
            the axis and distance difference.
        """
        if axis == 'X':
            # the endpoint shifts left or right from the center
            end_point = (
                self.center[0] + length
                if distance_diff > 0
                else self.center[0] - length,
                self.center[1]
            )
        elif axis == 'Y':
            # the endpoint shifts up or down from the center
            end_point = (
                self.center[0],
                self.center[1] + length
                if distance_diff > 0
                else self.center[1] - length
            )
        elif axis == 'Z':
            # the endpoint shifts diagonally from the center
            end_point = (
                self.center[0] + length
                if distance_diff > 0
                else self.center[0] - length,
                self.center[1] + length
                if distance_diff > 0
                else self.center[1] - length
            )
        return end_point

    def _normalize_tip_length(self, end_point: tuple) -> float:
        """
        Normalizes the tip length of the arrow based on the distance to the
        endpoint.

        Args:
            end_point (tuple): The coordinates of the arrow's endpoint.

        Returns:
            float: The normalized tip length, scaled based on the distance to
            the endpoint. Returns 0 if the distance is zero.
        """
        arrow_vec = np.array(self.center) - np.array(end_point)
        arrow_length = np.linalg.norm(arrow_vec)
        return self.fixed_tip_length / arrow_length if arrow_length > 0 else 0

    def show(
        self,
        image: np.ndarray,
        angle_difference: np.ndarray,
        distance_difference: np.ndarray,
    ) -> None:
        """
        Displays the image with highlighted angle and distance differences,
        and marks the center if both angle and distance differences are within
        tolerance.

        Args:
            image (np.ndarray): The image to be displayed.
            angle_difference (np.ndarray): The angle differences for X, Y, and
            Z axes.
            distance_difference (np.ndarray): The distance differences for X,
            Y, and Z axes.

        Returns:
            None
        """
        self.img = image
        self._draw_template()

        self._highlight_angle('X', angle_difference[0])
        self._highlight_angle('Y', angle_difference[1])
        self._highlight_angle('Z', angle_difference[2])

        self._highlight_distance('X', distance_difference[0])
        self._highlight_distance('Y', distance_difference[1])
        self._highlight_distance('Z', distance_difference[2])

        within_angle_tolerance = np.all(
            np.abs(angle_difference) <= self.tolerance
        )
        within_distance_tolerance = np.all(
            np.abs(distance_difference) <= self.tolerance
        )

        if within_angle_tolerance and within_distance_tolerance:
            cv.circle(self.img, self.center, 5, self._RGB['Y'], -1)


def position_chessboard(
    video_capture: cv.VideoCapture,
    camera_matrix: np.ndarray,
    distortion_matrix: np.ndarray,
    corners_ref: np.ndarray,
    distances_ref: np.ndarray,
    angles_ref: np.ndarray,
    configuration: Settings,
) -> None:
    """
    Estimates the position and orientation of a chessboard pattern in real-time
    from a video stream, and displays relevant information such as the position
    differences, measurements, and navigation data.

    Args:
        video_capture (cv.VideoCapture): The video capture object to get frame
        from the camera.
        camera_matrix (np.ndarray): The camera intrinsic matrix.
        distortion_matrix (np.ndarray): The camera distortion coefficients.
        corners_ref (np.ndarray): Reference points for the corners of the
        chessboard.
        distances_ref (np.ndarray): Reference distances for the chessboard.
        angles_ref (np.ndarray): Reference angles for the chessboard.
        configuration (Settings): Configuration settings for the calibration
        and display.

    Key Press Options:
        'A' : Toggle additional info display.
        'C' : Toggle chessboard corner display.
        'F' : Flip the image vertically.
        'I' : Save the current image.
        'M' : Toggle measurement display.
        'N' : Toggle navigation ball (direction finder) display.
        ESC : Exit the process.

    Returns:
        None
    """
    img_flipped = False
    show_add_info, show_corn, show_meas, show_nav = False, True, True, False

    dist, angl_euler = np.full(3, np.nan), np.full(3, np.nan)
    dist_diff, angl_eul_diff = np.full(3, np.nan), np.full(3, np.nan)

    objp = generate_object_points(configuration)
    axis = generate_3d_axis(configuration.SQUARE_SIZE)

    _, img = video_capture.read()
    nav_ball = NavigationBall(
        img,
        configuration.NAV_BALL_WDGT_SIZE,
        configuration.NAV_BALL_CLR,
        configuration.NAV_BALL_BCKGRND_CLR,
        configuration.MEAS_TOL,
        configuration.DIST_UNIT,
        configuration.ANGLE_UNIT,
    )

    menu_description = {
        'A': 'Additional Info',
        'C': 'Corners',
        'F': 'Flip Image',
        'I': 'Save Image',
        'M': 'Measurements',
        'N': 'Navigation',
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

            _, rvecs, tvecs = cv.solvePnP(
                objp, corn_det, camera_matrix, distortion_matrix,
            )

            dist = tvecs_to_xyz(tvecs)
            angl_euler = rvecs_to_xyz_euler(rvecs)

            dist_diff = dist - distances_ref
            angl_eul_diff = compute_angle_difference(angl_euler, angles_ref)

            if show_corn:
                draw_corners(img, corners_ref, configuration.RED)
                draw_corners(img, corn_det, configuration.BLUE)

                if is_within_tolerance(
                    dist_diff, angl_eul_diff, configuration.MEAS_TOL,
                ):
                    draw_corners(img, corners_ref, configuration.GREEN)

            if show_add_info:
                img_points, _ = cv.projectPoints(
                    axis, rvecs, tvecs, camera_matrix, distortion_matrix,
                )
                draw_axis_3d(img, corn_det[0], img_points, configuration)

        if img_flipped:
            img = cv.rotate(img, cv.ROTATE_180)

        if show_meas:
            add_text_xyz(
                img, dist_diff, angl_eul_diff, configuration, 'Difference', 800
            )
            add_text_xyz(img, dist, angl_euler, configuration, 'Actual', 0)
            add_text_xyz(
                img,
                distances_ref,
                angles_ref,
                configuration,
                'Reference',
                400,
            )

        if show_nav:
            nav_ball.show(img, angl_eul_diff, dist_diff)

        if show_add_info:
            draw_principal_point(img, camera_matrix)

        add_menu(img, menu_description, configuration.WHITE)

        cv.imshow(configuration.CB_PATT, img)

        if k == 27:
            # <escape> key to exit
            print('Leaving position estimation process...\n')
            break
        elif k == ord('a'):
            show_add_info = not show_add_info
        elif k == ord('c'):
            show_corn = not show_corn
        elif k == ord('f'):
            img_flipped = not img_flipped
        elif k == ord('i'):
            img_name = generate_file_name('pose', 'jpg', configuration.CB_PATT)
            cv.imwrite(img_name, img)
        elif k == ord('m'):
            show_meas = not show_meas
        elif k == ord('n'):
            show_nav = not show_nav


def position_menu(
    camera_index: int,
    camera_matrix: np.ndarray,
    distortion_matrix: np.ndarray,
    corners_ref: np.ndarray,
    distances_ref: np.ndarray,
    angles_ref: np.ndarray,
    configuration: Settings,
) -> None:
    """
    Displays a menu for initiating the chessboard position estimation process
    via camera feed. Allows the user to start the position estimation or exit
    the menu.

    Args:
        camera_index (int): Index of the camera to be used for position
        estimation.
        camera_matrix (np.ndarray): The camera intrinsic matrix.
        distortion_matrix (np.ndarray): The camera distortion coefficients.
        corners_ref (np.ndarray): Reference points for the corners of the
        chessboard.
        distances_ref (np.ndarray): Reference distances for the chessboard.
        angles_ref (np.ndarray): Reference angles for the chessboard.
        configuration (Settings): Configuration settings for the camera and
        display.

    Key Press Options:
        'P' : Start the position estimation process for the chessboard.
        ESC : Exit the menu.

    Returns:
        None
    """
    menu_description = {'P': 'Position', 'ESC': 'Escape'}

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
            elif k == ord('p'):
                position_chessboard(
                    cap,
                    camera_matrix,
                    distortion_matrix,
                    corners_ref,
                    distances_ref,
                    angles_ref,
                    configuration,
                )

        cap.release()
        cv.destroyAllWindows()
        cv.waitKey(1)
