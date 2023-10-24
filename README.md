# Traffic-Counting-in-Real-Time
Traffuc Counting in Real-Time using live video stream/IP camera in OpenCV.

> NOTE: This is an improvement/modification to https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/

[tracking.webm](https://github.com/chingizseyidbayli/RealTimeTrafficTracking/assets/15063807/4a6543d9-a488-4f78-a1f0-c3bd2150d0d3)


--- 

## Simple Theory

### SSD detector

- We are using a SSD ```Single Shot Detector``` with a MobileNet architecture. In general, it only takes a single shot to detect whatever is in an image. That is, one for generating region proposals, one for detecting the object of each proposal. 
- Compared to other two shot detectors like R-CNN, SSD is quite fast.
- ```MobileNet```, as the name implies, is a DNN designed to run on resource constrained devices. For e.g., mobiles, ip cameras, scanners etc.
- Thus, SSD seasoned with a MobileNet should theoretically result in a faster, more efficient object detector.

### Centroid tracker

- Centroid tracker is one of the most reliable trackers out there.
- To be straightforward, the centroid tracker computes the ```centroid``` of the bounding boxes.
- That is, the bounding boxes are ```(x, y)``` co-ordinates of the objects in an image. 
- Once the co-ordinates are obtained by our SSD, the tracker computes the centroid (center) of the box. In other words, the center of an object.
- Then an ```unique ID``` is assigned to every particular object deteced, for tracking over the sequence of frames.

---

## Running Inference

### Install the dependencies

First up, install all the required Python dependencies by running: ```
pip install -r requirements.txt ```

> NOTE: Supported Python version is 3.11.3 (there can always be version conflicts between the dependencies, OS, hardware etc.).

### Test video file

To run inference on a test video file, head into the root directory and run the command: 

```
python people_counter.py --prototxt detector/MobileNetSSD_deploy.prototxt --model detector/MobileNetSSD_deploy.caffemodel --input utils/data/tests/test_1.mp4
```




## References

***Main:***

- SSD paper: https://arxiv.org/abs/1512.02325
- MobileNets paper: https://arxiv.org/abs/1704.04861
- Centroid tracker: https://www.pyimagesearch.com/2018/07/23/simple-object-tracking-with-opencv/
- https://github.com/saimj7/People-Counting-in-Real-Time
