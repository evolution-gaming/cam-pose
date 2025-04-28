"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

import numpy as np

from src.utils import is_within_tolerance


def test_all_below_tolerance(sample_settings):
    dist_diff = np.array([0.1, 0.2, 0.3])
    angl_diff = np.array([0.05, 0.1, 0.2])
    assert is_within_tolerance(
        dist_diff, angl_diff, sample_settings.MEAS_TOL
    ) is np.True_


def test_all_above_tolerance(sample_settings):
    dist_diff = np.array([0.6, 0.7, 0.8])
    angl_diff = np.array([0.6, 0.7, 0.8])
    assert is_within_tolerance(
        dist_diff, angl_diff, sample_settings.MEAS_TOL
    ) is np.False_


def test_mixed_below_and_above_tolerance(sample_settings):
    dist_diff = np.array([0.1, 0.6, 0.3])
    angl_diff = np.array([0.05, 0.6, 0.2])
    assert is_within_tolerance(
        dist_diff, angl_diff, sample_settings.MEAS_TOL
    ) is np.False_


def test_exactly_at_tolerance(sample_settings):
    dist_diff = np.array([0.5, 0.5, 0.5])
    angl_diff = np.array([0.5, 0.5, 0.5])
    assert is_within_tolerance(
        dist_diff, angl_diff, sample_settings.MEAS_TOL
    ) is np.False_


def test_edge_case_zero_difference(sample_settings):
    dist_diff = np.array([0.0, 0.0, 0.0])
    angl_diff = np.array([0.0, 0.0, 0.0])
    assert is_within_tolerance(
        dist_diff, angl_diff, sample_settings.MEAS_TOL
    ) is np.True_


def test_edge_case_with_nonzero_differences(sample_settings):
    dist_diff = np.array([0.49, 0.51, 0.2])
    angl_diff = np.array([0.49, 0.51, 0.1])
    assert is_within_tolerance(
        dist_diff, angl_diff, sample_settings.MEAS_TOL
    ) is np.False_
