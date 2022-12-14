import argparse
import os
import time
import numpy as np
import math

import cv2
import depthai as dai

from utils import FPS, Timer, SliderWindow
from daipipeline import get_model_list, add_pipeline_args, create_pipeline
from poseestimators import get_poseestimator, add_poseestimator_args

model_list = get_model_list()

def parse_arguments():
    """
    Define the command line arguments for choosing a model etc.

    Returns
    -------
    args object
    """
    parser = argparse.ArgumentParser(description='')
    add_pipeline_args(parser, model_list)
    add_poseestimator_args(parser)
    parser.add_argument('-v',
                        '--video',
                        type=str,
                        help="If given, run on the video rather than oak "
                        "camera.")
    return vars(parser.parse_args())

def main(args):
    """
    Main programm loop.

    Parameters
    ----------
    args : command line arguments parsed by parse_arguments
    """
    # Setup PoseEstimator, pipeline, windows with sliders for PoseEstimator
    # options and load video if running on local video file
    camera = args["video"] is None
    if args["model"] not in model_list:
        raise ValueError("Unknown model '{}'".format(args["model"]))
    model_config = model_list[args["model"]]
    pose_estimator = get_poseestimator(model_config, **args)

    # Create Depth Pipeline
    pipeline = create_pipeline(model_config, camera, passthrough=True, **args)

    # Define sources and outputs
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoRight = pipeline.create(dai.node.MonoCamera)
    depth = pipeline.create(dai.node.StereoDepth)

    xout_dpth = pipeline.create(dai.node.XLinkOut)
    xout_dpth.setStreamName("depth_mm")

    # Properties
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

    # Create a node that will produce the depth map (using disparity output as 
    # it's easier to visualize depth this way)
    depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

    # Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
    depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
    depth.setLeftRightCheck(True)
    depth.setExtendedDisparity(False)
    depth.setSubpixel(False)

    # Linking
    monoLeft.out.link(depth.left)
    monoRight.out.link(depth.right)
    depth.depth.link(xout_dpth.input)

    with dai.Device(pipeline) as device:
        device.startPipeline()

        dpth_q = device.getOutputQueue(name="depth_mm", maxSize=4,\
         blocking=False)

        if camera:
            preview_queue = device.getOutputQueue("preview", maxSize=4,\ 
             blocking=False)
        else:
            pose_in_queue = device.getInputQueue("pose_in")
        pose_queue = device.getOutputQueue("pose")
        passthrough_queue = device.getOutputQueue("passthrough")

        # Load video if given in command line and set the variables used below
        # to control FPS and looping of the video
        if not camera:
            if not os.path.exists(args["video"]):
                raise ValueError("Video '{}' does not exist.".format(
                    args["video"]))
            print("Loading video", args["video"])
            video = cv2.VideoCapture(args["video"])
            frame_interval = 1 / video.get(cv2.CAP_PROP_FPS)
            last_frame_time = 0
            frame_id = 0
        else:
            print("Running on OAK camera preview stream")

        # Create windows for the original video and the video of frames from
        # the NN passthrough. The window for the original video gets all the
        # option sliders to change pose estimator config
        video_window_name = "Original Video"
        passthrough_window_name = "Processed Video"
        video_window = SliderWindow(video_window_name)
        cv2.namedWindow(passthrough_window_name)
        video_window.add_poseestimator_options(pose_estimator, args)

        # Start main loop
        frame = None
        keypoints = None
        fps = FPS("Video", "NN", interval=0.1)
        timer = Timer("inference", "decode")
        while True:
            # Check for and handle slider changes
            slider_changes = video_window.get_changes()
            for option_name, value in slider_changes.items():
                pose_estimator.set_option(option_name, value)

            fps.start_frame()
            # Get next video frame (and submit for processing if local video)
            if camera:
                frame = preview_queue.get().getCvFrame()
                fps.count("Video")
            else:
                frame_time = time.perf_counter()
                # Only grab next frame from file at certain intervals to
                # roughly preserve its original FPS
                if frame_time - last_frame_time > frame_interval:
                    if video.grab():
                        __, frame = video.retrieve()
                        fps.count("Video")
                        last_frame_time = frame_time
                        # Create DepthAI ImgFrame object to pass to the
                        # camera
                        input_frame = pose_estimator.get_input_frame(frame)
                        frame_nn = dai.ImgFrame()
                        frame_nn.setSequenceNum(frame_id)
                        frame_nn.setWidth(input_frame.shape[2])
                        frame_nn.setHeight(input_frame.shape[1])
                        frame_nn.setType(dai.RawImgFrame.Type.BGR888p)
                        frame_nn.setFrame(input_frame)
                        pose_in_queue.send(frame_nn)
                        frame_id += 1
                    else:
                        frame_id = 0
                        video.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

            # Process pose data whenever a new packet arrives
            if pose_queue.has():
                raw_output = pose_queue.get()

                # get depth queue
                inDepth = dpth_q.get()
                # get depth frame
                dpth_frame = inDepth.getFrame()

                timer.start_timer("decode")
                keypoints = pose_estimator.get_pose_data(raw_output)
                timer.stop_timer("decode")
                fps.count("NN")
                # When keypoints are available we should also have a
                # passthrough frame to process and display. Make sure it is
                # availabe to avoid suprises.
                if passthrough_queue.has():
                    passthrough = passthrough_queue.get()
                    timer.frame_time("inference", passthrough)
                    passthrough_frame = passthrough.getCvFrame()
                    passthrough_frame = pose_estimator.get_original_frame(
                        passthrough_frame)
                    pose_estimator.draw_results(keypoints, passthrough_frame)

                    # Draw Rectangle around keypoint, label it with landmark 
                    # and get depth value, take average of the rectangle 
                    # around each keypoint and display it

                    # if specific keypoint exists on screen
                    if keypoints.size != 0:
                        # loop through every keypoint, which is in format 
                        # [xVal, yVal, confidence]
                        for idx in range(len(keypoints[0])):
                            # if confidence is higher than 0 
                            # the point exists on screen
                            if keypoints[0][idx][2] != 0 and \
                             pose_estimator.landmarks[idx] == "right ankle":
                                currentPoint = keypoints[0][idx]
                                # covert preview frame pixel to depth frame
                                # float value is ratio between both
                                # preview is 456x256 and depth is 640x400
                                xVal = int(currentPoint[0] * 1.403508)
                                if xVal > 629:
                                    xVal = 629
                                yVal = int(currentPoint[1] * 1.5625)
                                if yVal > 389:
                                    yVal = 389
                                offset = 10

                                # Make and display the bounding box and label 
                                # of the keypoint we are observing
                                cv2.rectangle(passthrough_frame, \
                                 (xVal - offset, yVal - offset),\
                                 (xVal + offset, yVal + offset),\
                                 (255, 0, 0), 2)
                                cv2.putText(passthrough_frame,\ 
                                 pose_estimator.landmarks[idx],\ 
                                 (xVal - offset, yVal - (offset*2)),\
                                 cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0))

                                # calculate and display the average depth 
                                # within the bounding box (20x20)
                                builtArr = np.empty((0,2), int)
                                for idY in range(yVal - offset, yVal + offset):
                                    builtArr = np.append(builtArr, \
                                     dpth_frame[[[idY]],\
                                     [range(xVal - offset, xVal + offset)]])

                                builtArr = builtArr[builtArr != 0]

                                if (builtArr.size != 0):
                                    avgDepth = int(np.average(builtArr))
                                    cv2.putText(passthrough_frame,\
                                     str(avgDepth),\ 
                                     (xVal - offset, yVal + (offset*2)),\
                                     cv2.FONT_HERSHEY_TRIPLEX, 0.5, (255, 0, 0))

                                print("xVal: " + str(xVal)\
                                 + ", yVal: " + str(400 - yVal))
                                print("Distance is: "\
                                 + str(np.average(builtArr) / 1000))
                                print(pose_estimator.landmarks[idx]\
                                 + calculatePoint( 50, 72, xVal, yVal,\
                                 (np.average(builtArr) / 1000) ))

                    cv2.imshow(passthrough_window_name, passthrough_frame)

            # Annotate current video frame with keypoints and FPS
            if keypoints is not None:
                pose_estimator.draw_results(keypoints, frame)
            fps.update()
            fps.display(frame)

            cv2.imshow(video_window_name, frame)

            if cv2.waitKey(1) == ord("q"):
                break
        fps.print_totals()
        timer.print_times()
        cv2.destroyAllWindows()

#calculate the point in world space given 
def calculatePoint(vfov, hfov, pxVal, pyVal, dist):
    cpx = 320
    cpy = 200
    cameraOrigin = [0, -3, 1]
    cameraOriginPhi = 108.435 
    cameraOriginTheta = 90
    tanVfovInRadians = math.tan(math.radians(vfov / 2))
    tanHfovInRadians = math.tan(math.radians(hfov / 2))

    #calculate phi and theta from camera center to picture point
    pixelPhiDeg = cameraOriginPhi - math.degrees(math.atan(\
     tanVfovInRadians * ( (((cpy*2) - pyVal) - cpy) / cpy) ))
    pixelThetaDeg = cameraOriginTheta - math.degrees(math.atan(\
     tanHfovInRadians * ((pxVal - cpx) / cpx) ))
    pixelPhiRad = math.radians(pixelPhiDeg)
    pixelThetaRad = math.radians(pixelThetaDeg)

    #calculate unit length vector
    xVal = math.sin(pixelPhiRad) * math.cos(pixelThetaRad)
    yVal = math.sin(pixelPhiRad) * math.sin(pixelThetaRad)
    zVal = math.cos(pixelPhiRad)

    #calculate world point
    objXVal = cameraOrigin[0] + (xVal * dist)
    objYVal = cameraOrigin[1] + (yVal * dist)
    objZVal = cameraOrigin[2] + (zVal * dist)

    print("Phi is: "+str(pixelPhiDeg) +" and Theta is: "+str(pixelThetaDeg))
    return " is in World Coordinate: (" + str(objXVal)\
     + ", " + str(objYVal) + ", " + str(objZVal) + ")"


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
