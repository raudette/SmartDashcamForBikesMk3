#!/usr/bin/env python3
# Running the yolov5s blob based on this code from Luxonis
# https://github.com/luxonis/depthai-experiments/tree/e12d6a7e2f40d3ada35c03fb7b0176b33efe960b/gen2-yolov5
# Running the depth perception based on this code from Luxonis
# https://github.com/luxonis/depthai-experiments/tree/master/gen2-calc-spatials-on-host

import cv2
import depthai as dai
from util.functions import non_max_suppression
import argparse
import time
import numpy as np
from calc import HostSpatialsCalc
import math

#settings
fps=5

labelMap = [
    "License_Plate"
]

parser = argparse.ArgumentParser()
parser.add_argument("-conf", "--confidence_thresh", help="set the confidence threshold", default=0.3, type=float)
parser.add_argument("-iou", "--iou_thresh", help="set the NMS IoU threshold", default=0.4, type=float)

args = parser.parse_args()

nn_path = 'SDCMk3-LicensePlateModel.blob'
conf_thresh = args.confidence_thresh
iou_thresh = args.iou_thresh

nn_shape = 416

def draw_boxes(frame, boxes, total_classes):
    if boxes.ndim == 0:
        return frame
    else:

        # define class colors
        colors = boxes[:, 5] * (255 / total_classes)
        colors = colors.astype(np.uint8)
        colors = cv2.applyColorMap(colors, cv2.COLORMAP_HSV)
        colors = np.array(colors)

        for i in range(boxes.shape[0]):
            x1, y1, x2, y2 = int(boxes[i,0]), int(boxes[i,1]), int(boxes[i,2]), int(boxes[i,3])
            conf, cls = boxes[i, 4], int(boxes[i, 5])

            label = f"{labelMap[cls]}: {conf:.2f}" if "default" in nn_path else f"Class {cls}: {conf:.2f}"
            color = colors[i, 0, :].tolist()

            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            # Get the width and height of label for bg square
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)

            # Shows the text.
            frame = cv2.rectangle(frame, (x1, y1 - 2*h), (x1 + w, y1), color, -1)
            frame = cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,255), 1)
    return frame


# Start defining a pipeline
pipeline = dai.Pipeline()

pipeline.setOpenVINOVersion(version = dai.OpenVINO.VERSION_2021_1)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(nn_path)
detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

cam = pipeline.createColorCamera()
cam.setPreviewSize(nn_shape,nn_shape)
cam.setInterleaved(False)
cam.preview.link(detection_nn.input)
cam.setFps(fps)

# Create outputs
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("nn_input")
xout_rgb.input.setBlocking(False)

detection_nn.passthrough.link(xout_rgb.input)
xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
xout_nn.input.setBlocking(False)
detection_nn.out.link(xout_nn.input)

# Depth cam setup
# Define sources and outputs
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)
# Properties
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
stereo.initialConfig.setConfidenceThreshold(255)
stereo.setLeftRightCheck(True)
stereo.setSubpixel(False)
# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("depth")
stereo.depth.link(xoutDepth.input)
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutDepth.setStreamName("disp")
stereo.disparity.link(xoutDepth.input)


# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_nn_input = device.getOutputQueue(name="nn_input", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    start_time = time.time()
    counter = 0
    fps = 0
    layer_info_printed = False
    while True:
        in_nn_input = q_nn_input.get()
        in_nn = q_nn.get()

        frame = in_nn_input.getCvFrame()

        layers = in_nn.getAllLayers()

        # get the "output" layer
        output = np.array(in_nn.getLayerFp16("output"))

        # reshape to proper format
        cols = output.shape[0]//10647
        output = np.reshape(output, (10647, cols))
        output = np.expand_dims(output, axis = 0)

        total_classes = cols - 5

        depthQueue = device.getOutputQueue(name="depth")
        dispQ = device.getOutputQueue(name="disp")
        hostSpatials = HostSpatialsCalc(device)
        y = 200
        x = 300
        step = 3
        delta = 5
        hostSpatials.setDeltaRoi(delta)

        boxes = non_max_suppression(output, conf_thres=conf_thresh, iou_thres=iou_thresh)
        boxes = np.array(boxes[0])

        if boxes is not None:
            frame = draw_boxes(frame, boxes, total_classes)
        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
        cv2.imshow("nn_input", frame)

        in_depthFrame = depthQueue.get()
        depthFrame = in_depthFrame.getFrame()
        print("depth frame")
        print(in_depthFrame.getTimestamp())
        print(in_depthFrame.getSequenceNum())


        # Calculate spatial coordiantes from depth frame
        spatials, centroid = hostSpatials.calc_spatials(depthFrame, (x,y)) # centroid == x/y in our case

        # Get disparity frame for nicer depth visualization
        disp = dispQ.get().getFrame()
        disp = (disp * (255 / stereo.initialConfig.getMaxDisparity())).astype(np.uint8)
        disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

        #text.rectangle(disp, (x-delta, y-delta), (x+delta, y+delta))
        #text.putText(disp, "X: " + ("{:.1f}m".format(spatials['x']/1000) if not math.isnan(spatials['x']) else "--"), (x + 10, y + 20))
        #text.putText(disp, "Y: " + ("{:.1f}m".format(spatials['y']/1000) if not math.isnan(spatials['y']) else "--"), (x + 10, y + 35))
        #text.putText(disp, "Z: " + ("{:.1f}m".format(spatials['z']/1000) if not math.isnan(spatials['z']) else "--"), (x + 10, y + 50))

        # Show the frame
        cv2.imshow("depth", disp)

        counter += 1
        if (time.time() - start_time) > 1:
            fps = counter / (time.time() - start_time)

            counter = 0
            start_time = time.time()

        if cv2.waitKey(1) == ord('q'):
            break
