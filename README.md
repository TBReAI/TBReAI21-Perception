# TBReAI21 Perception
### Computationally Efficient Perception for Autonomous Racing

Building the perception pipeline for Team Bath Racing Electric's 2021 autonomous racing car. The code for the 2019/2020 implementation of the perception pipeline, which was based on the YOLO v3-Tiny algorithm and built using C++, can be found [**here**](YOLOv3-Tiny-Implementation/code).

The video below displays how our current model (i.e. the 2019/2020 implementation) detects and classifies blue and yellow cones for the purpose of autonomous racing.  

![alt text](https://github.com/TBReAI/TBReAI21-Perception/blob/main/YOLOv3-Tiny-Implementation/images-and-video/old-detection.gif "Detection GIF")

The current goal is to improve upon the work of the past year by building a more computationally-efficient perception model that is capable of achieving higher detection accuracy with lower latency. Our hope is to deploy the system on our vehicle for the [**2021 FS-AI competition**](https://www.imeche.org/events/formula-student/team-information/fs-ai). We will be building this new model using Python to be more aligned with the other systems in our autonomous pipeline. 

##### UPDATE
> Currently working on gaining access to the FSOCO dataset.
##### UPDATE
> Until we gain access to the FSOCO dataset for our 2021 implementation, we will be using the MIT driverless open source dataset.
##### UPDATE
> Cloned training code for YOLOv3 in PyTorch from [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3). 
