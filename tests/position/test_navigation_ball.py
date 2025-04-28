"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

import numpy as np
import pytest


def test_initialization(nav_ball):
    # assert
    assert np.array_equal(nav_ball.img,
                          np.zeros((500, 500, 3), dtype=np.uint8))
    assert nav_ball.widget_size == 500
    assert nav_ball.ball_color == (150, 150, 150)
    assert nav_ball.background_color == (0, 0, 0)
    assert nav_ball.tolerance == 0.5
    assert nav_ball.dist_unit == 'mm'
    assert nav_ball.angl_unit == 'deg'


def test_draw_template(nav_ball, sample_image):
    # arrange
    nav_ball.img = sample_image.copy()

    # act
    nav_ball._draw_template()

    # assert the template modifies the image
    assert not np.all(nav_ball.img == 0)


def test_highlight_angle(nav_ball, sample_image):
    # arrange
    nav_ball.img = sample_image.copy()

    # act
    nav_ball._highlight_angle('X', 10.0)

    # assert the method modifies the image
    assert not np.all(nav_ball.img == 0)


def test_highlight_angle_nan(nav_ball, sample_image):
    # arrange
    nav_ball.img = sample_image.copy()

    # act
    nav_ball._highlight_angle('X', np.nan)

    # assert the method don't modifies the image
    assert not np.all(nav_ball.img != 0)


def test_highlight_distance(nav_ball, sample_image):
    # arrange
    nav_ball.img = sample_image.copy()

    # act
    nav_ball._highlight_distance('Y', 20.0)

    # assert the method modifies the image
    assert not np.all(nav_ball.img == 0)


def test_highlight_distance_nan(nav_ball, sample_image):
    # arrange
    nav_ball.img = sample_image.copy()

    # act
    nav_ball._highlight_distance('Y', np.nan)

    # assert the method don't modifies the image
    assert not np.all(nav_ball.img != 0)


@pytest.mark.parametrize('axis, angle_diff, expected_type', [
    ('X', 15.0, tuple),
    ('Y', -10.0, tuple),
    ('Z', 0.0, tuple),
])
def test_calculate_arrow_position(nav_ball, axis, angle_diff, expected_type):
    # act
    arrow_tip, arrow_end = nav_ball._calculate_arrow_position(axis, angle_diff)

    # assert
    assert isinstance(arrow_tip, expected_type)
    assert isinstance(arrow_end, expected_type)


@pytest.mark.parametrize('axis, distance_diff, length, expected_type', [
    ('X', 15.0, 50, tuple),
    ('Y', -10.0, 30, tuple),
    ('Z', 5.0, 70, tuple),
])
def test_calculate_distance_point(
    nav_ball, axis, distance_diff, length, expected_type,
):
    # act
    end_point = nav_ball._calculate_distance_point(axis, distance_diff, length)

    # assert
    assert isinstance(end_point, expected_type)


def test_normalize_tip_length_normal_case(nav_ball):
    # arrange
    end_point = (300, 300)

    # act
    tip_length = nav_ball._normalize_tip_length(end_point)
    expected_arrow_length = np.linalg.norm(
        np.array(nav_ball.center) - np.array(end_point)
    )
    expected_tip_length = nav_ball.fixed_tip_length / expected_arrow_length

    # assert
    msg = 'Tip length calculation failed for normal case'
    assert tip_length == pytest.approx(expected_tip_length), msg


def test_normalize_tip_length_zero_length_case(nav_ball):
    # arrange
    end_point = (260, 240)

    # act
    tip_length = nav_ball._normalize_tip_length(end_point)

    # assert
    msg = 'Tip length should be 0 for zero-length arrow'
    assert tip_length == 0, msg


def test_show(nav_ball, sample_image):
    # arrange
    angle_difference = np.array([10.0, -5.0, 20.0])
    distance_difference = np.array([15.0, -10.0, 5.0])

    # act
    nav_ball.show(sample_image.copy(), angle_difference, distance_difference)

    # assert the method modifies the image
    assert not np.all(nav_ball.img == 0)
