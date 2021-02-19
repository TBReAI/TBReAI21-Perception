### Requirements for each program to compile:

- LiDAR requires  [ OPENCV, PCL ]
- STEREO requires [ OPENCV, ZED] , ZED requires CUDA.

### Implementation steps:

All resources, such as the neural network weights and configuration filesm are located in /build/res/

For LiDAR, the pcd point cloud (pcd_read) located
in /build/res.

Run on linux using: `./pcd_read < .pcd file >`

Some testing pcd files are included in res/

As the training data is in RGB, the CNN only detects LiDAR information with RGB values not intensity.

For Stereo, the config and weights for YOLOV3-tiny all located in /build/res/

To compile the individual code, navigate to the build folder in a linux terminal for each program and use: `cmake .. && make`

To compile the code:

Run LiDAR with: `./perception <.pcd file>`

Run STEREO with: `./perception`

> If no ZED cam is found, any webcam built into the device will be used.

