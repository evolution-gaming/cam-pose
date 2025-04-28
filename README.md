# Camera Position Estimation Tool (cam-pose)

## About
`cam-pose` is a tool designed to estimate the position of a camera in 3D space. By analyzing input data, it computes the camera's orientation and location relative to a defined coordinate system (reference template).

This functionality is essential in various applications, including computer vision, robotics, and augmented reality, where understanding the camera's viewpoint enhances scene interpretation and interaction.

## Dependencies
The `cam-pose` project relies on [Python](https://www.python.org/) version >= 3.11 and its several packages to function correctly. These packages are specified in the [requirements.txt](requirements.txt) file.

## Install
Ensure you have [Python](https://www.python.org/) installed.

Clone the repository to your local machine using the command:
<pre>git clone "project_path"</pre>

Change directory:
<pre>cd cam-pose</pre>

Create virtual environment by running:
<pre>virtualenv venv</pre>

Activate virtual environment with:
<pre>source venv/bin/activate</pre>

Install the required dependencies by running:
<pre>pip install -r requirements.txt</pre>

## Documentation
For usage documentation and examples head to [usage documentation](doc/usage.md).

## License
`cam-pose` tool source code is licensed under [MIT license](https://choosealicense.com/licenses/mit/), see [LICENSE](LICENSE).

This project uses third-party dependencies that are licensed under MIT, BSD, Apache 2.0, Python Software Foundation (PSF), and MIT-CMU licenses. See the respective package documentation for more information.
