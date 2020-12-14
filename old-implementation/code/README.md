### Requirements for each program to compile:

LiDAR requires  [ OPENCV, PCL ]
STEREO requires [ OPENCV, ZED] , ZED requires CUDA.


All the resources such as the neural network weights and configuration files are located in each program: folder /build/res/

For LiDAR, there is a pcd point cloud reader called pcd_read located
in the /build/res folder. run on linux using: 

./pcd_read < .pcd file >

Some testing pcd files are included in the res/ folder.

As the training data is in RGB, the CNN only detects LiDAR information with RGB values not intensity.

For Stereo, the config and weights for YOLOV3-tiny all located
in /build/res/

To compile the individual code, ( The compiled executable is also include in the build directory)
Navigate to the build folder in a linux terminal for each program
use 
    "cmake .. && make"

To compile the code:

run LiDAR with: ./perception <.pcd file>
    Stereo    : ./perception 
    ( if no zed cam is found, it will use any webcam 
    built into your device. However, no distance approximation)

