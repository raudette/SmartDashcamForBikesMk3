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
import gpsd
import threading
from multiprocessing import Process, Queue, current_process
import queue # imported for using queue.Empty exception

#defaults/constants
manualfocus=100 #0 to 255
camerafps=2
recordingfolder="./data/"
loggpspositionintervalsec=10
number_of_ocr_processes = 2 #we want 1 less than number of cores
ocr_tasks_to_accomplish = Queue()
ocr_tasks_that_are_done = Queue()
max_ocr_queuesize=5 #just start dropping frames if we can't keep up
processes = []

DBlock = threading.Lock()

def do_ocr_job(ocr_tasks_to_accomplish, ocr_tasks_that_are_done):
    ocr = PaddleOCR(use_angle_cls=True, lang='en') 
    #silence paddleocr
    logger = logging.getLogger('root')
    logger.setLevel(logging.WARN)
    while True:
        try:
            task = ocr_tasks_to_accomplish.get_nowait()
            #not sure what the overhead of this is, but probably not thread safe, so we'll put it here
            result = ocr.ocr(task[0], cls=True)
            for index, line in enumerate(result):
                if line[1][1]>=args['ocr_confidence']:
                    #remove whitespace and punctuation that is sometimes detected in place of the Ontario crown symbol
                    sPlate=''.join(filter(str.isalnum,str(line[1][0])))
                    fConfidence=str(line[1][1])
                    sPaddleResult="paddle " + str(line[1][0]) + " confidence " + str(fConfidence)
                    print(sPaddleResult)
                    runsql("INSERT INTO feature (routelogid,featureplatenumber, featuretypeid, feature_x, feature_y, feature_z,timestamp,devicetimestamp) VALUES ('"+str(task[1])+"','"+sPlate+"',1,"+str(task[2])+","+str(task[3])+","+str(task[4])+",'"+task[5]+"','"+str(task[6])+"')")

        except queue.Empty:
            pass
        except KeyboardInterrupt:
            break

def decode(nn_data: NNData):
    layer = nn_data.getFirstLayerFp16()
    results = np.array(layer).reshape((1, 1, -1, 7))
    dets = Detections(nn_data)

    for result in results[0][0]:
        if result[2] > 0.5:
            dets.add(result[1], result[2], result[3:])

    return dets

def loggpsposition():
    threading.Timer(loggpspositionintervalsec, loggpsposition).start()
    now = datetime.now()
    pctime=now.strftime("%Y-%m-%d %H:%M:%f")
    runsql(buildroutelogsql(pctime))

#have several threads updating DB - ensure only one is using the DB connection
def runsql(sql):
    try:
        DBlock.acquire(True)
        cursor.execute(sql)
        lastrowid=cursor.lastrowid
        dbconnection.commit()
    finally:
        DBlock.release()
    return lastrowid


#we have 2 places we want to log GPS coordinates, so I break out the required SQL
def buildroutelogsql(pctime):
    velocity=0 #for ANT+ sensor eventually
    mode=0 #no gps or gps not locked
    print(pctime)
    if args['gps']:
        try:
            gpspacket = gpsd.get_current()
            mode = gpspacket.mode 
        except: #if no GPS device connected
            pass
    if mode == 2: #2d fix
        routelogquery="INSERT INTO routelog (sessionid,timestamp, latitude, longitude, velocity, gpstimestamp, gpsvelocity, latitude_err, longitude_err, gpsvelocity_err) VALUES ('"+str(sessionId)+"','"+pctime+"','"+str(gpspacket.lat)+"','"+str(gpspacket.lon)+"','"+str(velocity)+"','"+gpspacket.get_time().strftime("%Y-%m-%d %H:%M:%f")+"','"+str(gpspacket.speed())+"','"+str(gpspacket.error['y'])+"','"+str(gpspacket.error['x'])+"','"+str(gpspacket.error['s'])+"')"
    elif mode > 2: #3d fix
        routelogquery="INSERT INTO routelog (sessionid,timestamp, latitude, longitude, velocity, gpstimestamp, gpsvelocity, latitude_err, longitude_err, gpsvelocity_err, altitude, altitude_err) VALUES ('"+str(sessionId)+"','"+pctime+"','"+str(gpspacket.lat)+"','"+str(gpspacket.lon)+"','"+str(velocity)+"','"+gpspacket.get_time().strftime("%Y-%m-%d %H:%M:%f")+"','"+str(gpspacket.speed())+"','"+str(gpspacket.error['y'])+"','"+str(gpspacket.error['x'])+"','"+str(gpspacket.error['s'])+"','"+str(gpspacket.lat)+"','"+str(gpspacket.error['v'])+"')"
    else: #no gps or no fix
        routelogquery = "INSERT INTO routelog (sessionid,timestamp,velocity) VALUES ('"+str(sessionId)+"','"+pctime+"','"+str(velocity)+"')"
    return routelogquery

def callback(packet: DetectionPacket, visualizer: Visualizer):
    detections: Detections = packet.img_detections

    num = len(packet.img_detections.detections)
    if num>=1:
        now = datetime.now()
        detecttime=now.strftime("%Y-%m-%d %H:%M:%f")
        routelogid = runsql(buildroutelogsql(detecttime))
        devicetimestamp=detections.getTimestampDevice()
    if args['desktop']:
        imgPreview = packet.frame
        if num<1:
            cv2.imshow("Augmented Output", imgPreview)

    for detectedobject in packet.img_detections.detections:
        roiData = detectedobject.boundingBoxMapping
        roi = roiData.roi
        roi = roi.denormalize(packet.frame.shape[1],packet.frame.shape[0])
        topLeft = roi.topLeft()
        bottomRight = roi.bottomRight()
        imgPlate = packet.frame[int(topLeft.y):int(bottomRight.y), int(topLeft.x):int(bottomRight.x)  ]
        if ( ocr_tasks_to_accomplish.qsize() <= max_ocr_queuesize):
            print("Adding to queue")
            ocr_tasks_to_accomplish.put([imgPlate,routelogid,detectedobject.spatialCoordinates.x,detectedobject.spatialCoordinates.y,detectedobject.spatialCoordinates.z,detecttime,devicetimestamp])
#                    runsql("INSERT INTO feature (routelogid,featureplatenumber, featuretypeid, feature_x, feature_y, feature_z,
# timestamp,devicetimestamp) VALUES ('"+str(routelogid)+"','"+sPlate+"',1,"+str(detectedobject.spatialCoordinates.x)+","+str(detectedobject.spatialCoordinates.y)+","+str(detectedobject.spatialCoordinates.z)+",'"+detecttime+"','"+str(devicetimestamp)+"')")

        else:
            print("OCR QUEUE Full, skipping detection")
        if args['desktop']:
            imgPreview = packet.frame
            cv2.rectangle(imgPreview, (int(topLeft.x),int(topLeft.y)),(int(bottomRight.x),int(bottomRight.y)),(0,255,0),2)
        if args['desktop']:
            cv2.imshow("Augmented Output", imgPreview)

parser = argparse.ArgumentParser()
parser.add_argument("-conf", "--config", help="Trained YOLO json config path", default='model/SDCMk3-LicensePlateModel.json', type=str)
parser.add_argument("-iou", "--iou_thresh", help="set the NMS IoU threshold", default=0.4, type=float)
parser.add_argument("-ocr", "--ocr_confidence", help="set the NMS IoU threshold", default=0.9, type=float)
parser.add_argument("-norec", "--norecording", help="Disable video", action='store_true')
parser.add_argument("-dt", "--desktop", help="Turn on things you want running while testing on desktop", action='store_true')
parser.add_argument("-gps", "--gps", help="Enable GPS", action='store_true')

args = ArgsParser.parseArgs(parser)

now = datetime.now()

dbconnection = sqlite3.connect("SmartDashcamForBikesMk3.sqlite3", check_same_thread=False)
cursor = dbconnection.cursor()

startTime = now.strftime("%Y%m%d-%H%M%S")
#cursor.execute("INSERT INTO session (description) VALUES ('"+startTime+"')")
sessionId=runsql("INSERT INTO session (description) VALUES ('"+startTime+"')")#cursor.lastrowid
#dbconnection.commit()

if args['gps']:
    gpsd.connect()
    loggpsposition() #starts a thread to log position at regular intervals

# creating ocr processes
for w in range(number_of_ocr_processes):
    p = Process(target=do_ocr_job, args=(ocr_tasks_to_accomplish, ocr_tasks_that_are_done))
    processes.append(p)
    p.start()

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
    color = oak.create_camera('color',fps=camerafps,encode='H265')
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
