#!/usr/bin/env python3
# Running the yolov5s blob based on this code from Luxonis
# https://github.com/luxonis/depthai-experiments/tree/master/gen2-yolo/device-decoding

from depthai_sdk import OakCamera, ArgsParser, Detections, NNData, DetectionPacket, Visualizer, RecordType
import argparse
import numpy as np
import cv2 as cv2
from paddleocr import PaddleOCR
import logging
import depthai as dai
import sqlite3
from datetime import datetime
import time
import os

#defaults/constants
manualfocus=100 #0 to 255
recordingfolder="./data/"

ocr = PaddleOCR(use_angle_cls=True, lang='en') 
#silence paddleocr
logger = logging.getLogger('root')
logger.setLevel(logging.WARN)

def decode(nn_data: NNData):
    layer = nn_data.getFirstLayerFp16()
    results = np.array(layer).reshape((1, 1, -1, 7))
    dets = Detections(nn_data)

    for result in results[0][0]:
        if result[2] > 0.5:
            dets.add(result[1], result[2], result[3:])

    return dets

def callback(packet: DetectionPacket, visualizer: Visualizer):
    detections: Detections = packet.img_detections

    now = datetime.now()
    detecttime=now.strftime("%Y-%m-%d %H:%M:%f")

    num = len(packet.img_detections.detections)
    if args['desktop']:
        imgPreview = packet.frame
        if num<1:
            cv2.imshow("Augmented Output", imgPreview)

    for detectedobject in packet.img_detections.detections:
        cursor.execute("INSERT INTO routelog (sessionid,timestamp) VALUES ('"+str(sessionId)+"','"+detecttime+"')")
        dbconnection.commit()
        routelogid=cursor.lastrowid
        print("x ", str(detectedobject.spatialCoordinates.x) + ",y " + str(detectedobject.spatialCoordinates.y) + ",z " + str(detectedobject.spatialCoordinates.z) )
        roiData = detectedobject.boundingBoxMapping
        roi = roiData.roi
        roi = roi.denormalize(packet.frame.shape[1],packet.frame.shape[0])
        topLeft = roi.topLeft()
        bottomRight = roi.bottomRight()
        imgPlate = packet.frame[int(topLeft.y):int(bottomRight.y), int(topLeft.x):int(bottomRight.x)  ]
        if args['desktop']:
            imgPreview = packet.frame
            cv2.rectangle(imgPreview, (int(topLeft.x),int(topLeft.y)),(int(bottomRight.x),int(bottomRight.y)),(0,255,0),2)
        result = ocr.ocr(imgPlate, cls=True)
        for index, line in enumerate(result):
            if line[1][1]>=args['ocr_confidence']:
                #remove whitespace and punctuation that is sometimes detected in place of the Ontario crown symbol
                sPlate=''.join(filter(str.isalnum,str(line[1][0])))
                fConfidence=str(line[1][1])
                sPaddleResult="paddle " + str(line[1][0]) + " confidence " + str(fConfidence)
                print(sPaddleResult)
                cursor.execute("INSERT INTO feature (routelogid,featureplatenumber, featuretypeid, feature_x, feature_y, feature_z,timestamp) VALUES ('"+str(routelogid)+"','"+sPlate+"',1,"+str(detectedobject.spatialCoordinates.x)+","+str(detectedobject.spatialCoordinates.y)+","+str(detectedobject.spatialCoordinates.z)+",'"+detecttime+"')")
                dbconnection.commit()
                if args['desktop']:
                    cv2.putText(imgPreview,sPaddleResult,(int(topLeft.x),int(topLeft.y)+20) , cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 255, 0))
        if args['desktop']:
            cv2.imshow("Augmented Output", imgPreview)

parser = argparse.ArgumentParser()
parser.add_argument("-conf", "--config", help="Trained YOLO json config path", default='model/SDCMk3-LicensePlateModel.json', type=str)
parser.add_argument("-iou", "--iou_thresh", help="set the NMS IoU threshold", default=0.4, type=float)
parser.add_argument("-ocr", "--ocr_confidence", help="set the NMS IoU threshold", default=0.9, type=float)
parser.add_argument("-norec", "--norecording", help="Disable video", action='store_true')
parser.add_argument("-dt", "--desktop", help="Turn on things you want running while testing on desktop", action='store_true')

args = ArgsParser.parseArgs(parser)

now = datetime.now()

dbconnection = sqlite3.connect("SmartDashcamForBikesMk3.sqlite3")
cursor = dbconnection.cursor()

startTime = now.strftime("%Y%m%d-%H%M%S")
cursor.execute("INSERT INTO session (description) VALUES ('"+startTime+"')")
sessionId=cursor.lastrowid
dbconnection.commit()

# All this to guess the folder that the video is going to go in
mxid=""
for device in dai.Device.getAllAvailableDevices():
    mxid=device.getMxId()

folders = os.listdir(recordingfolder)
recordingId=0
for folder in folders:
    index = folder.index("-") if "-" in folder else -1 
    if index >= 1:
        tempId = folder[0:folder.index("-")]
        if tempId.isnumeric:
            if int(tempId) > recordingId:
                recordingId=int(tempId)
recordingId = recordingId +1
sVideoName = str(recordingId)+"-"+mxid+"/color.mp4"
print(sVideoName)


with OakCamera(args=args) as oak:
    color = oak.create_camera('color',fps=2,encode='H265')
    nn = oak.create_nn(args['config'], color, nn_type='yolo', spatial=True)
    color.node.initialControl.setManualFocus(manualfocus)
    color.node.initialControl.SceneMode(dai.CameraControl.SceneMode.SPORTS)
    color.node.initialControl.setAutoFocusMode(dai.RawCameraControl.AutoFocusMode.OFF)
    if not args['norecording']:
        oak.record([color.out.encoded], recordingfolder, RecordType.VIDEO)
        cursor.execute("UPDATE session set videofile='"+sVideoName+"' where sessionid = "+str(sessionId)) 
        dbconnection.commit()
    oak.visualize(nn, callback=callback)
    oak.start(blocking=True)
