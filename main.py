"""
Copyright ©2025 Evolution. All rights reserved.
Use of this source code is governed by a MIT-style
license that can be found in the LICENSE file.
"""

from src.calibration import calibration_menu
from src.config import config
from src.position import position_menu
from src.reference import reference_menu
from src.utils import (detect_camera_indexes, display_saved_data, load_data,
                       select_camera_index, select_saved_data)


def main() -> None:
    """
    Runs the main menu loop for a camera calibration and positioning system.

    This function provides a command-line interface to manage camera-related
    tasks, including selecting a camera index, performing calibration, setting
    reference data, and determining the camera position. The user is presented
    with a menu to choose various options, and the function ensures that the
    necessary data is selected before executing relevant operations.

    Menu Options:
    - 1: Select camera index
    - 2: Positioning (requires calibration and reference data)
    - 3: Reference data selection (requires calibration data)
    - 4: Calibration menu
    - 5: Detect available camera indexes
    - 0: Exit the program

    The function runs indefinitely until the user selects option '0' to exit.

    Preconditions:
    - Some operations require a valid camera index to be selected first.
    - Positioning requires both calibration and reference data.
    - Reference menu requires calibration data.

    Raises:
    - Prints error messages if required data is missing.

    Returns:
    - None
    """
    cam_idx, asterisks = -1, '* ' * 30
    while True:
        print(f'\n{asterisks}')
        print('Menu:')
        print('1 - Select camera index')
        print('2 - Position')
        print('3 - Reference')
        print('4 - Calibration')
        print('5 - Detect camera indexes')
        print('0 - Exit')

        choice = input('\nEnter your choice: ')
        print(asterisks)

        if choice == '1':
            cam_idx = select_camera_index()

        elif choice == '2':
            if cam_idx >= 0:
                saved_data = display_saved_data()
                if saved_data:
                    print('Please select calibration data\n')
                    selected_cal_data = select_saved_data(saved_data)
                    cal_data = load_data(selected_cal_data)
                    if len(cal_data) == 4:
                        cam_mtx, dist_coeffs, _, _ = cal_data
                        print('Please select reference data\n')
                        saved_data = display_saved_data()
                        selected_ref_data = select_saved_data(saved_data)
                        ref_data = load_data(selected_ref_data)
                        if len(ref_data) == 3:
                            corn_ref, dist_ref, angl_ref = ref_data
                            position_menu(
                                cam_idx,
                                cam_mtx,
                                dist_coeffs,
                                corn_ref,
                                dist_ref,
                                angl_ref,
                                config,
                            )
                        else:
                            print('Reference data must be of shape 3')
                    else:
                        print('Calibration data must be of shape 4')
            else:
                print('Select camera index first')

        elif choice == '3':
            if cam_idx >= 0:
                saved_data = display_saved_data()
                if saved_data:
                    print('Please select calibration data\n')
                    selected_cal_data = select_saved_data(saved_data)
                    cal_data = load_data(selected_cal_data)
                    if len(cal_data) == 4:
                        cam_mtx, dist_coeffs, _, _ = cal_data
                        reference_menu(cam_idx, cam_mtx, dist_coeffs, config)
                    else:
                        print('Calibration data must be of shape 4')
            else:
                print('Select camera index first')

        elif choice == '4':
            if cam_idx >= 0:
                calibration_menu(cam_idx, config)
            else:
                print('Select camera index first')

        elif choice == '5':
            print(f'\nDetected camera indexes: {detect_camera_indexes()}')

        elif choice == '0':
            print('Exiting...')
            break

        else:
            print('Invalid input')


if __name__ == '__main__':
    main()
