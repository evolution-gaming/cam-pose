"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

from unittest.mock import patch

from src.calibration import delete_points


@patch('builtins.input', return_value='2')
def test_delete_existing_index(mock_input, sample_calibration_set):
    # arrange
    sample_obj_points, sample_image_points, _ = sample_calibration_set

    # act
    is_deleted = delete_points(sample_obj_points, sample_image_points)

    # assert
    msg = 'Expected True result for existing index'
    assert is_deleted is True, msg
    msg = 'Expected one entry to be deleted'
    assert len(sample_obj_points) == 2 and len(sample_image_points) == 2, msg


@patch('builtins.input', return_value='5')
def test_out_of_range_index(mock_input, sample_calibration_set):
    # arrange
    sample_obj_points, sample_image_points, _ = sample_calibration_set

    # act
    is_deleted = delete_points(sample_obj_points, sample_image_points)

    # assert
    msg = 'Expected False result for out-of-range index'
    assert is_deleted is False, msg
    msg = 'Expected no deletion for out-of-range index'
    assert len(sample_obj_points) == 3 and len(sample_image_points) == 3, msg


@patch('builtins.input', return_value='abc')
def test_non_integer_input(mock_input, sample_calibration_set):
    # arrange
    sample_obj_points, sample_image_points, _ = sample_calibration_set

    # act
    is_deleted = delete_points(sample_obj_points, sample_image_points)

    # assert
    msg = 'Expected False result for non-integer input'
    assert is_deleted is False, msg
    msg = 'Expected no deletion for non-integer input'
    assert len(sample_obj_points) == 3 and len(sample_image_points) == 3, msg


@patch('builtins.input', return_value='1')
def test_delete_last_index(mock_input, sample_calibration_set):
    # arrange
    sample_obj_points, sample_image_points, _ = sample_calibration_set
    del sample_obj_points[1:]
    del sample_image_points[1:]

    # act
    is_deleted = delete_points(sample_obj_points, sample_image_points)

    # assert
    msg = 'Expected True result for the last index'
    assert is_deleted is True, msg
    msg = 'Expected one entry to be deleted'
    assert len(sample_obj_points) == 0 and len(sample_image_points) == 0, msg
