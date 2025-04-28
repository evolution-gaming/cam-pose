"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

from unittest.mock import patch

from src.utils import detect_camera_indexes


def test_no_cameras_available(sample_video_capture):
    # mock
    with patch('cv2.VideoCapture') as MockVideoCapture:
        MockVideoCapture.return_value.isOpened.return_value = False

        # act
        result = detect_camera_indexes()

        # assert
        msg = 'Expected an empty list when no cameras are available'
        assert result == [], msg


def test_single_camera_available():
    # mock
    with patch('cv2.VideoCapture') as MockVideoCapture:
        MockVideoCapture.return_value.isOpened.side_effect = lambda: MockVideoCapture.call_args[0][0] == 0  # noqa

        # act
        result = detect_camera_indexes()

        # assert
        msg = (
            'Expected a list with only index 0 when '
            'only one camera is available'
        )
        assert result == [0], msg


def test_multiple_cameras_available():
    # mock
    with patch('cv2.VideoCapture') as MockVideoCapture:
        MockVideoCapture.return_value.isOpened.side_effect = lambda: MockVideoCapture.call_args[0][0] in {0, 1, 3}  # noqa

        # act
        result = detect_camera_indexes()

        # assert
        msg = 'Expected indexes [0, 1, 3] for multiple cameras available'
        assert result == [0, 1, 3], msg


def test_high_index_cameras_available():
    # mock
    with patch('cv2.VideoCapture') as MockVideoCapture:
        MockVideoCapture.return_value.isOpened.side_effect = lambda: MockVideoCapture.call_args[0][0] in {7, 8}  # noqa

        # act
        result = detect_camera_indexes()

        # assert
        msg = (
            'Expected indexes [7, 8] when cameras are '
            'available at high indexes'
        )
        assert result == [7, 8], msg


def test_video_capture_release(sample_video_capture):
    # mock
    with patch('cv2.VideoCapture') as MockVideoCapture:
        MockVideoCapture.return_value.isOpened.side_effect = lambda: MockVideoCapture.call_args[0][0] == 0  # noqa
        mock_instance = MockVideoCapture.return_value

        # act
        detect_camera_indexes()

        # assert
        msg = (
            'Expected release() to be called exactly once for each open camera'
        )
        assert mock_instance.release.call_count == 1, msg
