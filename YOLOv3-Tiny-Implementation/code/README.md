### Requirements for each program to compile

- LiDAR requires  [ OPENCV, PCL ]
- STEREO requires [ OPENCV, ZED] , ZED requires CUDA.

### Details

#### LiDAR

For LiDAR, the pcd point cloud (pcd_read) is located in [lidar/build/res/](lidar/build/res)

- Run on linux using: `./pcd_read < .pcd file >`

Some pcd testing files are included [here](lidar/build/res).

As the training data is in RGB, the CNN only detects LiDAR information with RGB values not intensity.

#### STEREO

For STEREO, the config and weights for YOLOv3-Tiny are all located in in [stereo/build/res/](stereo/build/res)

#### Implementation steps

To compile the code, navigate to the build folder in a linux terminal for each program and use: `cmake .. && make`

- Run LiDAR with: `./perception <.pcd file>`

- Run STEREO with: `./perception`

> If no ZED cam is found, any webcam built into the device will be used.

