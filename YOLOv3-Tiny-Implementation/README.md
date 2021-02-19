# Multisensory Perception Pipeline

The pipeline uses multiple sensory inputs to evaluate the environment features.

### Stereo Cameras
A ZED Stereo Camera will be used to gather the information in the field of view. The left input frames will be used to both detect and classify the cones by using a YOLOv3-tiny object detector. For each detected cone, the bounding box of the cone is determined using the detector. The location estimate of the detected cones can be obtained by using an HSV filter to extract the correct cone color from the bounding box. The final cone distance estimation will be produced by averaging several point clouds from the extracted color pixels. 

### 3D LiDAR
A 3D LiDAR will be used to provide redundancy to the stereo camera. This is done by extracting the same information from the environment. When a point cloud is acquired, the backview of the point cloud is initially trimmed. Next, using a threshold-based filter, the ground plane in the point cloud can be removed. The remaining information is then obtained by using the Euclidean Clustering algorithm. Each cluster is projected into a 2D plane to create an image to feed into a convolutional neural network, which then classifies the image. Finally the centroid of each cluster is calculated to provide a final estimate location. 
