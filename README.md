# TrafficAnalysis 
Fully automated camera calibration using 3 orthogonal vanishing points and scene scale from 3-D bounding boxes from detected vehicles (Faster-RCNN) to estimate speed of vehicles.

# Getting Started 
This repo contains 3 main folders: 
- `Data` : This contains images and videos used during this project. 
- `Notebooks` : This contains different notebooks and python scripts used to create 3d bounding box, estimated vanishing points and estimate vehicle speed after faster-rcnn detection and kalman filter vehicle tracking.
- `Results` : This contains final video showing estimated 3d bounding box around detected and tracked vehicles. 

**Further Description of Notebooks/Scripts**

`Notebooks` folder contains the following:
- `VP_DiamondSpace.ipynb`: This explains the process of finding vanishing points using Diamond Space Accumulator
- `VP_RANSAC_frame.ipynb`: This explains the process of finding vanishing points using RANSAC method on a single frame of a video
- `VP_RANSAC_KLTframe.ipynb`: This explains the process of finding vanishing points using RANSAC method by tracking the moving vehicles using KLT on a video.
- `construct_3dbb.ipynb`: This defines the method of constructing 3d bounding box from a given set of 3 vanishing points and 2d vehicle detection masks by using 2 frames of a single car at different distances from camera.
- `speed_noTracker1.ipynb`: This uses the Mask RCNN object detector from Tensorflow Hub to detect a car in a initial frame of a video and use the method from construct_3dbb.ipynb to estimated 3d bounding box and scene scale and estimate the speed of car wrt a final frame. 



# Requirements/Installations


# Steps


# Issues


# References


# LICENSE

