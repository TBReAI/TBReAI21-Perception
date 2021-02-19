# TBReAI21 Perception
### Computationally Efficient Perception for Autonomous Racing

Building the perception pipeline for Team Bath Racing Electric's 2021 autonomous racing car. The code for the current implementation of the perception pipeline, which was based on the YOLO v3-Tiny network structure and built using C++, can be found [**here**](YOLOv3-Tiny-Implementation/code). A LiDAR-based detector was also built using convolutional neural networks. However, as our vehicle will only utilize stereo cameras, this detector will be scrapped for future iterations. 

The video below displays how our current model detects and classifies blue and yellow cones for the purpose of autonomous racing.  

![alt text](https://github.com/TBReAI/TBReAI21-Perception/blob/main/YOLOv3-Tiny-Implementation/images-and-video/old-detection.gif "Detection GIF")

The current goal is to improve upon the work of the past year by building a more computationally-efficient perception model that is capable of achieving higher detection accuracy with lower latency. Our hope is to deploy the system on our vehicle for the [**2021 FS-AI competition**](https://www.imeche.org/events/formula-student/team-information/fs-ai). We will be building this new model using Python to be more aligned with the other systems in our autonomous pipeline. PyTorch, used by Tesla to develop full self-driving capabilities for their vehicles, will be the library utilized for our pipeline. 

##### NOTICE
> Updated code will be made available upon completion of the project.
