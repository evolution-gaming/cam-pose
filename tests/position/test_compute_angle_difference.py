"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

from src.position import compute_angle_difference


def test_small_positive_difference():
    assert compute_angle_difference(10.0, 5.0) == 5.0


def test_small_negative_difference():
    assert compute_angle_difference(5.0, 10.0) == -5.0


def test_wraparound_positive():
    assert compute_angle_difference(350.0, 5.0) == -15.0


def test_wraparound_negative():
    assert compute_angle_difference(5.0, 350.0) == 15.0


def test_exact_180_difference_negative():
    assert compute_angle_difference(0.0, 180.0) == -180.0


def test_exact_180_difference_negative_swap():
    assert compute_angle_difference(180.0, 0.0) == -180.0


def test_near_360_wraparound():
    assert compute_angle_difference(0.0, 359.0) == 1.0


def test_near_360_wraparound_negative():
    assert compute_angle_difference(359.0, 0.0) == -1.0


def test_180_degree_flip_negative():
    assert compute_angle_difference(90.0, 270.0) == -180.0
