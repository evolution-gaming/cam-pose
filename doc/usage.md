# Usage of Camera Position Estimation Tool (cam-pose)

## Patterns
For accurate camera calibration, use structured patterns like the checkerboard pattern. Pattern can be generated using online tools like [calib.io](https://calib.io/pages/camera-calibration-pattern-generator).

## Configuration
To customize the behavior of the `cam-pose` tool, you can modify the [config.env](../config.env) file located in the project's root directory. This file contains various parameters that control aspects of the tool's operation. Below is a breakdown of the configurable parameters:

**Configuration Parameters:**

- **Window Settings:**
  - `WINDOW_WIDTH`: Sets the width of the display window in pixels.
  - `WINDOW_HEIGHT`: Sets the height of the display window in pixels.

- **Camera Settings:**
  - `FRAME_WIDTH`: Specifies the width of the camera frame in pixels.
  - `FRAME_HEIGHT`: Specifies the height of the camera frame in pixels.

- **Object Settings:**
  - `CB_ROWS`: Defines the number of rows in the checkerboard pattern used for camera positioning.
  - `CB_COLUMNS`: Defines the number of columns in the checkerboard pattern used for camera positioning.
  - `SQUARE_SIZE`: Indicates the size of each square in the checkerboard pattern, measured in millimeters.

- **Tolerances:**
  - `MEAS_TOL`: Sets the measurement tolerance, which could be used to define acceptable error margins in measurements.

- **Units:**
  - `DIST_UNIT`: Specifies the unit of distance measurements (e.g., 'mm' for millimeters).
  - `ANGLE_UNIT`: Specifies the unit of angle measurements (e.g., 'deg' for degrees).

- **Direction Finder:**
  - `NAV_BALL_WDGT_SIZE`: Specifies the widget size of direction finder in pixels.
  - `NAV_BALL_CLR`: Specifies the main color of direction finder (e.g., [150, 150, 150] for grey).
  - `NAV_BALL_BCKGRND_CLR`: Specifies the background color of direction finder (e.g., [0, 0, 0] for black).

After making any changes to the [config.env](../config.env) file, ensure you restart the tool to apply the new configurations.

By adjusting these parameters, you can tailor the tool to better suit your specific requirements and operating environment.

## Usage Example
Below is a step-by-step guide to using the `cam-pose` tool for camera calibration and position estimation.

- **Setup and Launch**
    - Prepare a checkerboard pattern.
    - Follow the installation instructions in [README.md](../README.md), then start the program:
<pre>python3 main.py</pre>
- **Detect Available Camera Sources**
    - Press [5] in the Main Menu to list available camera indices.
- **Select Camera Source**
    - Press [1] in the Main Menu to select the camera.
    - Based on the detected sources, enter the corresponding camera index.
- **Camera Calibration**
    - Open the Calibration Menu (press [4] in the Main Menu).
    - Position the calibration pattern in front of the camera.
    - Start collecting calibration data: press [C].
    - Stop data collection and initiate calibration: press [Esc].
    - Visualize and analyze results: press [V].
    - If needed, delete specific data points by entering their index: press [D].
    - Save the calibration matrix: press [S].
    - Return to the Main Menu: press [Esc].
- **Reference Position Calculation**
    - Open the Reference Menu (press [3] in the Main Menu).
    - Select a previously saved calibration file.
    - Position the calibration pattern in front of the camera.
    - Start reference position calculation: press [R].
    - Save the reference position: press [S].
    - Stop the process: press [Esc].
    - Return to the Main Menu: press [Esc].
- **Deviation Calculation from Reference Position**
    - Open the Position Menu (press [2] in the Main Menu).
    - Select the previously saved calibration file.
    - Select the previously saved reference position file.
    - Position the calibration pattern in front of the camera.
    - Start camera positioning: press [P].
    - After positioning, stop the process: press [Esc].
    - Return to the Main Menu: press [Esc].
- **Exit the Program**
    - Press [0] in the Main Menu to exit.

## Testing
Run tests and linters to ensure the tool functions correctly:
<pre>./run_tests.sh</pre>

This script performs the following actions:
* Runs Unit Tests: Utilizes the [pytest](https://pypi.org/project/pytest/) framework to execute tests located in the tests directory, verifying the correctness of the codebase.
* Executes Linters: Employs tools like [flake8](https://pypi.org/project/flake8/) to analyze the code for stylistic errors and adherence to coding standards.

Regularly running this script helps maintain code quality and functionality throughout the development process.
