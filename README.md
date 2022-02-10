<!-- Add banner here -->
<!-- ![Banner]() -->

# DepthSliceTool

<!-- Add buttons here -->

![GitHub last commit](https://img.shields.io/github/last-commit/zeinhajjali/DepthSliceTool)
![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/zeinhajjali/DepthSliceTool?include_prereleases)
![GitHub issues](https://img.shields.io/github/issues/zeinhajjali/DepthSliceTool)

<!-- Describe your project in brief -->

A tool for automatically segmenting videos based on depth data. Used for region-of-interest segmentation and selection for videos of neonatal patients in the NICU. Developed as part of a project by the [Carleton University - Biomedical Informatics Co-laboratory (CU-BIC)](https://cu-bic.ca/).

Features include:

- Head and torso region-of-interest segmentation
- Perspective transformation of recorded depth video
- Compatibility with rosbag (.bag) files recorded by Intel RealSense RGB-D cameras
- Combining and encoding rosbag files to mj2 archival video files.

![Demo GIF](staticFiles/demo.gif)

# Table of contents

- [DepthSliceTool](#depthslicetool)
- [Table of contents](#table-of-contents)
- [Installation](#installation)
- [Usage](#usage)
- [Development](#development)
- [Contribute](#contribute)
- [License](#license)

# Installation
[(Back to top)](#table-of-contents)

The tools were tested and confirmed to run with Python 3.8.10. The scripts to encode rosbag files as mp4 or mj2 videos run in MATLAB R2019a. You can install the python dependencies for the tools from the 'requirements.txt' file.

Pip:
```
pip install -r requirements.txt
```
Conda:
```
conda install --file requirements.txt
```

The scripts found at 'bagmerge/rosbag_merge.py' and 'bagmerge/realsense_rosbag_merge.py' require an active ROS Noetic environment to run properly. Instructions to install on Ubuntu (or WSL) can be found [here](http://wiki.ros.org/noetic/Installation/Ubuntu).

# Usage
[(Back to top)](#table-of-contents)

WIP

# Development
[(Back to top)](#table-of-contents)

WIP

As of right now the development environment is the same as the one used to run the scripts.

# Contribute
[(Back to top)](#table-of-contents)

WIP

# License
[(Back to top)](#table-of-contents)

[WIP]

<!-- Add the footer here -->

<!-- ![Footer]() -->
