import cv2
import depthai as dai
import numpy as np

# Closer-in minimum depth, disparity range is doubled (from 95 to 190):
extended_disparity = False
# Better accuracy for longer distance, fractional disparity 32-levels:
subpixel = False
# Better handling for occlusions:
lr_check = True

# Create pipeline
pipeline = dai.Pipeline()

# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
depth = pipeline.create(dai.node.StereoDepth)

xout_dpth = pipeline.create(dai.node.XLinkOut)
xout_dpth.setStreamName("depth_mm")

xout_dsp = pipeline.create(dai.node.XLinkOut)
xout_dsp.setStreamName("disparity")

# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
depth.setLeftRightCheck(lr_check)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)

# Linking
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)
depth.disparity.link(xout_dsp.input)
depth.depth.link(xout_dpth.input)

color = (255, 0, 0)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    # Output queue will be used to get the disparity frames from the outputs defined above
    q = device.getOutputQueue(name="disparity", maxSize=4, blocking=False)

    # Output queue will be used to get the depth frames from the outputs defined above
    dpth_q = device.getOutputQueue(name="depth_mm", maxSize=4, blocking=False)

    while True:
        inDisparity = q.get()  # blocking call, will wait until a new data has arrived
        inDepth = dpth_q.get()

        #Shape Specified by Width, Height (400x600), so presumably frame[0][0] is top left pixel's depth
        #Left half = [0-400][0-300]
        frame = inDisparity.getFrame()
        dpth_frame = inDepth.getFrame()
d
        #Getting Starting pixels for our 5x5 Grid
        widthStep = inDisparity.getWidth() // 5
        heightStep = inDisparity.getHeight() // 5

        #Loop that calculates the avg of pixels in each quadrent in the 5x5
        for quadrent in range (0, 25):
            quadrentWidthStart = widthStep * (quadrent % 5)
            quadrentVertStart = heightStep * (quadrent // 5)
            
            #using python's wild indexing and numpy to build an array of slices of the main array
            #which whill then be averaged and displayed
            builtArr = np.empty((0,2), int)
            for idX in range(quadrentVertStart, quadrentVertStart + heightStep):
                builtArr = np.append(builtArr, dpth_frame[[[idX]], [range(quadrentWidthStart, quadrentWidthStart + widthStep)]])
            
            avgDepth = int(np.average(builtArr))

            cv2.putText(frame, str(avgDepth), (quadrentWidthStart + (widthStep // 2), quadrentVertStart + (heightStep // 2)), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color)

        # Normalization for better visualization
        frame = (frame * (255 / depth.initialConfig.getMaxDisparity())).astype(np.uint8)

        #print(dpth_frame)

        cv2.imshow("disparity", frame)

        # Available color maps: https://docs.opencv.org/3.4/d3/d50/group__imgproc__colormap.html
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        cv2.imshow("disparity_color", frame)

        if cv2.waitKey(1) == ord('q'):
            break