#!/usr/bin/env python3

import cv2
import depthai as dai
from util.functions import non_max_suppression
import argparse
import time
import numpy as np
import blobconverter
from calc import HostSpatialsCalc
from utility import *
import math


'''
YoloV5 object detector running on selected camera.
Run as:
python3 -m pip install -r requirements.txt
python3 main.py -cam rgb
Possible input choices (-cam):
'rgb', 'left', 'right'

Blob is taken from ML training examples:
https://github.com/luxonis/depthai-ml-training/tree/master/colab-notebooks

You can clone the YoloV5_training.ipynb notebook and try training the model yourself.

'''
labelMap = [
    "License_Plate"
]
cam_options = ['rgb', 'left', 'right']

parser = argparse.ArgumentParser()
parser.add_argument("-cam", "--cam_input", help="select camera input source for inference", default='rgb', choices=cam_options)
parser.add_argument("-nn", "--nn_model", help="select model path for inference", default='roboflow4.blob', type=str)
parser.add_argument("-conf", "--confidence_thresh", help="set the confidence threshold", default=0.3, type=float)
parser.add_argument("-iou", "--iou_thresh", help="set the NMS IoU threshold", default=0.4, type=float)

args = parser.parse_args()

cam_source = args.cam_input
nn_path = args.nn_model
conf_thresh = args.confidence_thresh
iou_thresh = args.iou_thresh

nn_shape = 416

def to_planar(frame):
    return frame.transpose(2, 0, 1).flatten()

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
version = "2021.1"
pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_1)

# Define a neural network that will make predictions based on the source frames
detection_nn = pipeline.createNeuralNetwork()
detection_nn.setBlobPath(nn_path)

detection_nn.setNumPoolFrames(4)
detection_nn.input.setBlocking(False)
detection_nn.setNumInferenceThreads(2)

cam=None
# Define a source - color camera
if cam_source == 'rgb':
    cam = pipeline.createColorCamera()
    cam.setPreviewSize(nn_shape,nn_shape)
    cam.setInterleaved(False)
    cam.preview.link(detection_nn.input)

#stereo cam setup
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


if cam_source != 'rgb':
    manip = pipeline.createImageManip()
    manip.setResize(nn_shape,nn_shape)
    manip.setKeepAspectRatio(True)
    manip.setFrameType(dai.RawImgFrame.Type.BGR888p)
    cam.out.link(manip.inputImage)
    manip.out.link(detection_nn.input)

cam.setFps(30)

# Create outputs
xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("nn_input")
xout_rgb.input.setBlocking(False)

detection_nn.passthrough.link(xout_rgb.input)

xout_nn = pipeline.createXLinkOut()
xout_nn.setStreamName("nn")
xout_nn.input.setBlocking(False)

detection_nn.out.link(xout_nn.input)


# ---------------------------------------
# 2nd stage NN - text-recognition-0012
# ---------------------------------------
#manip = pipeline.createImageManip()
#manip.setWaitForConfigInput(True)

#manip_img = pipeline.createXLinkIn()
#manip_img.setStreamName('manip_img')
#manip_img.out.link(manip.inputImage)

#manip_cfg = pipeline.createXLinkIn()
#manip_cfg.setStreamName('manip_cfg')
#manip_cfg.out.link(manip.inputConfig)

#manip_xout = pipeline.createXLinkOut()
#manip_xout.setStreamName('manip_out')

platein = pipeline.createXLinkIn()
platein.setStreamName('platein')

nn2 = pipeline.createNeuralNetwork()
#nn2.setBlobPath(blobconverter.from_zoo(name="text-recognition-0012", shaves=6, version=version))
nn2.setBlobPath("text-recognition-0012_openvino_2021.1_6shave.blob")
nn2.setNumInferenceThreads(2)
platein.out.link(nn2.input)


#manip.out.link(nn2.input)
#manip.out.link(manip_xout.input)

nn2_xout = pipeline.createXLinkOut()
nn2_xout.setStreamName("recognitions")
nn2.out.link(nn2_xout.input)


# Pipeline defined, now the device is assigned and pipeline is started
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    q_nn_input = device.getOutputQueue(name="nn_input", maxSize=4, blocking=False)
    q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    # and the text recognition bits here
    q_rec = device.getOutputQueue("recognitions", 4, blocking=True)
    #q_manip_img = device.getInputQueue("manip_img")
    #q_manip_cfg = device.getInputQueue("manip_cfg")
    #q_manip_out = device.getOutputQueue("manip_out", 4, blocking=False)
    q_platein = device.getInputQueue("platein")

    depthQueue = device.getOutputQueue(name="depth")
    dispQ = device.getOutputQueue(name="disp")
    text = TextHelper()
    hostSpatials = HostSpatialsCalc(device)
    y = 200
    x = 300
    step = 3
    delta = 5
    hostSpatials.setDeltaRoi(delta)


    class CTCCodec(object):
        """ Convert between text-label and text-index """
        def __init__(self, characters):
            # characters (str): set of the possible characters.
            dict_character = list(characters)

            self.dict = {}
            for i, char in enumerate(dict_character):
                self.dict[char] = i + 1

            self.characters = dict_character
            #print(self.characters)
            #input()
        def decode(self, preds):
            """ convert text-index into text-label. """
            texts = []
            index = 0
            # Select max probabilty (greedy decoding) then decode index to character
            preds = preds.astype(np.float16)
            preds_index = np.argmax(preds, 2)
            preds_index = preds_index.transpose(1, 0)
            preds_index_reshape = preds_index.reshape(-1)
            preds_sizes = np.array([preds_index.shape[1]] * preds_index.shape[0])

            for l in preds_sizes:
                t = preds_index_reshape[index:index + l]

                # NOTE: t might be zero size
                if t.shape[0] == 0:
                    continue

                char_list = []
                for i in range(l):
                    # removing repeated characters and blank.
                    if not (i > 0 and t[i - 1] == t[i]):
                        if self.characters[t[i]] != '#':
                            char_list.append(self.characters[t[i]])
                text = ''.join(char_list)
                texts.append(text)

                index += l

            return texts

    characters = '0123456789abcdefghijklmnopqrstuvwxyz#'
    codec = CTCCodec(characters)


    start_time = time.time()
    counter = 0
    fps = 0
    layer_info_printed = False
    #rotated_rectangles = []
    rec_pushed = 0
    rec_received = 0
    #q_nn_input.get().setTimestamp(time.monotonic())
    while True:
        is_there_q_rec = True
        while is_there_q_rec == True:
            in_rec = q_rec.tryGet()
            if in_rec is None:
                is_there_q_rec = False
            else:
                rec_data = bboxes = np.array(in_rec.getFirstLayerFp16()).reshape(30,1,37)
                decoded_text = codec.decode(rec_data)[0]
                #pos = rotated_rectangles[rec_received]
                print("{:2}: {:20}".format(rec_received, decoded_text),
                    "center({:3},{:3}) size({:3},{:3}) angle{:5.1f} deg".format(
                        int(pos[0][0]), int(pos[0][1]), pos[1][0], pos[1][1], pos[2]))
                rec_received += 1

        in_nn_input = q_nn_input.get()
        in_nn = q_nn.get()
        frame = in_nn_input.getCvFrame()
        print("input image")
        print(in_nn_input.getTimestamp().total_seconds())
        print(in_nn_input.getSequenceNum())
        print("nn")
        print(in_nn.getTimestamp().total_seconds())
        print(in_nn.getSequenceNum())
        layers = in_nn.getAllLayers()

        # get the "output" layer
        output = np.array(in_nn.getLayerFp16("output"))

        # reshape to proper format
        cols = output.shape[0]//10647
        output = np.reshape(output, (10647, cols))
        output = np.expand_dims(output, axis = 0)

        total_classes = cols - 5

        boxes = non_max_suppression(output, conf_thres=conf_thresh, iou_thres=iou_thresh)
        boxes = np.array(boxes[0])

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

        text.rectangle(disp, (x-delta, y-delta), (x+delta, y+delta))
        text.putText(disp, "X: " + ("{:.1f}m".format(spatials['x']/1000) if not math.isnan(spatials['x']) else "--"), (x + 10, y + 20))
        text.putText(disp, "Y: " + ("{:.1f}m".format(spatials['y']/1000) if not math.isnan(spatials['y']) else "--"), (x + 10, y + 35))
        text.putText(disp, "Z: " + ("{:.1f}m".format(spatials['z']/1000) if not math.isnan(spatials['z']) else "--"), (x + 10, y + 50))

        # Show the frame
        cv2.imshow("depth", disp)



        if boxes is not None:
            frame = draw_boxes(frame, boxes, total_classes)
            if boxes.ndim != 0:
                for i in range(boxes.shape[0]):
                    print("here")
                    x1, y1, x2, y2 = int(boxes[i,0]), int(boxes[i,1]), int(boxes[i,2]), int(boxes[i,3])
                    conf, cls = boxes[i, 4], int(boxes[i, 5])
                    
                    #cfg = dai.ImageManipConfig()
                    #cfg.setCropRotatedRect(rr, False)
                    #cfg.setResize(120, 32)
                    # Send frame and config to device
                    subframe=frame[y1:y2, x1:x2]
                    cv2.imshow("plate",subframe)
                    spatials, centroid = hostSpatials.calc_spatials(depthFrame, (x1,y1,x2,y2)) # centroid == x/y in our case
                    cv2.putText(frame, "Z: " + ("{:.1f}m".format(spatials['z']/1000) if not math.isnan(spatials['z']) else "--"), (x1 + 10, y1 + 50),cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
                    tstamp = time.monotonic()
                    lic_frame = dai.ImgFrame()
                    #lic_frame.setData(to_planar(subframe, (120, 32)))
                    lic_frame.setData(to_planar(subframe))
                    lic_frame.setTimestamp(tstamp)
                    #lic_frame.setSequenceNum(frame_det_seq)
                    lic_frame.setType(dai.RawImgFrame.Type.BGR888p)
                    w,h,c = subframe.shape
                    #lic_frame.setWidth(120)
                    #lic_frame.setHeight(32)
                    #w,h,c = subframe.shape
                    #imgFrame = dai.ImgFrame()
                    #imgFrame.setData(to_planar(subframe))
                    #imgFrame.setType(dai.ImgFrame.Type.BGR888p)
                    #imgFrame.setWidth(w)
                    #imgFrame.setHeight(h)
                    #q_platein.send(lic_frame)
                    print("here2")

        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, (255, 0, 0))
        cv2.imshow("nn_input", frame)

        counter += 1
        if (time.time() - start_time) > 1:
            fps = counter / (time.time() - start_time)

            counter = 0
            start_time = time.time()


        if cv2.waitKey(1) == ord('q'):
            break
