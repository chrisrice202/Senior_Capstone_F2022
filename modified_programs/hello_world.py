import numpy as np  # numpy - manipulate the packet data returned by depthai
import cv2  # opencv - display the video stream
import depthai  # depthai - access the camera and its data packets
import blobconverter  # blobconverter - compile and download MyriadX neural network blobs

# "Any action from DepthAI, whether it’s a neural inference or color camera output, 
# require a pipeline to be defined, including nodes and connections corresponding to our needs."
pipeline = depthai.Pipeline()

# creating a ColorCamera object that will access the color camera on the OAK-D
cam_rgb = pipeline.create(depthai.node.ColorCamera)
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setInterleaved(False)

detection_nn = pipeline.create(depthai.node.MobileNetDetectionNetwork)

# Set up a stereo depth node for use on dumping pixel depth data
# stereoDepth = pipeline.create(depthai.node.StereoDepth) 

# Set path of the blob (NN model). We will use blobconverter to convert&download the model
# detection_nn.setBlobPath("/path/to/model.blob")
detection_nn.setBlobPath(blobconverter.from_zoo(name='mobilenet-ssd', shaves=6))
detection_nn.setConfidenceThreshold(0.5)

# Connects the color camera's OUTPUT to the nueral network's INPUT
cam_rgb.preview.link(detection_nn.input)

# Uses XLINK - the thing that connects host to device and device to host
xout_rgb = pipeline.create(depthai.node.XLinkOut)
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_nn = pipeline.create(depthai.node.XLinkOut)
xout_nn.setStreamName("nn")
detection_nn.out.link(xout_nn.input)

labelMap = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
color = (255, 0, 0)

with depthai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb")
    q_nn = device.getOutputQueue("nn")
    frame = None
    detections = []

    #focal length calculations
    #focal_length_mm = focaLen * (6.17mm (sensor width) / 4056 (pixel width of sensor)
    #fl_mm = ( focal length (in pixels) * sensor width (in mm) ) / pixel width of sensor
    #sensor size is 1/4 inch, so the sensor size is 3.20mm (width) x 2.40mm (height)
    #fl_mm = (801.8759155273438 * 3.2) / 1280
    #fl_mm = 2.00468978882 which is ~2mm

    

    calibData = device.readCalibration()
    calibData.getBaselineDistance() 
    print(calibData.getCameraIntrinsics(depthai.CameraBoardSocket.RIGHT)[0][0])

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    while True:
        in_rgb = q_rgb.tryGet()
        in_nn = q_nn.tryGet()
        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
        if in_nn is not None:
            detections = in_nn.detections
        if frame is not None:

            for detection in detections:
                bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            cv2.imshow("preview", frame)
        if cv2.waitKey(1) == ord('q'):
            break
