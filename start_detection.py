import jetson.inference
import jetson.utils
from collections import deque

import imutils
import sys
import cv2
import time
import numpy as np
from device_controller import startDevice
from device_controller import stopDevice
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--confidence", type=float, default=0.3, help="Confidence threshold for person detection")
parser.add_argument("-s", "--stream", help="Video stream source")
parser.add_argument("-rt", "--startthreshold", type=int, default=5, help="Time to wait before starting device (seconds)")
parser.add_argument("-st", "--stopthreshold", type=int, default=20, help="Time to wait before stopping device (seconds)")
args = vars(parser.parse_args())

def init_model():
    # load the object detection network
    global net, class_names
    net = jetson.inference.detectNet("ssd-mobilenet-v2", sys.argv, 0.3)
    labels_file = open("models/class_labels.txt", "r")
    class_names = list(labels_file.read().split("\n"))
    labels_file.close()


def processDetection(frame, detections):
    detectionQueue = deque(maxlen=20)
    for detection in detections:
        detected_class = class_names[detection.ClassID]
        confidence = detection.Confidence
        if detected_class != "person" or confidence < float(args["confidence"]):
            detectionQueue.append(0)
            continue
        detectionQueue.append(1)
        left = int(detection.Left)
        right = int(detection.Right)
        width = int(detection.Width)
        height = int(detection.Height)
        top = int(detection.Top)
        bottom = int(detection.Bottom)
        center_x = int(detection.Center[0])
        center_y = int(detection.Center[1])

        # cv2.rectangle(frame, (top, left), (bottom, right), (0,255,0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
        # cv2.putText(frame, detected_class, (top-10, left-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (5,5), (int(0.5 * w), int(0.25 * h)), (0,0,0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "Human Activity: ", (int(0.03 * w), 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)
    cv2.putText(frame, "Device Status: ", (int(0.03 * w), 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)
    cv2.putText(frame, "Frames per second: ", (int(0.03 * w), 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)
    cv2.putText(frame, "Press 'Q' to exit", (int(0.03 * w), 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)

    if (np.array(detectionQueue).mean() >= 0.5):
        activityDetected = True
        cv2.putText(frame, "Detected", (int(0.03 * w) + 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 0), 1)

    else:
        activityDetected = False
        cv2.putText(frame, "Not Detected", (int(0.03 * w) + 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 255), 1)

    return frame, activityDetected


if args["stream"] in ['0','1','2','3','4','5']:
    stream = cv2.VideoCapture(int(args["stream"]))
else:
    stream = cv2.VideoCapture(args["stream"])

init_model()

activityDetected = False
waitTimer = None
timeLeft = None
startThreshold = args["startthreshold"]
stopThreshold = args["stopthreshold"]

while True:
    isRead, frame = stream.read()
    h, w = frame.shape[:2]

    if isRead:
        fps_start = time.time()
        width = frame.shape[1]
        height = frame.shape[0]
        img = jetson.utils.cudaFromNumpy(frame)
        detections = net.Detect(img, width, height, overlay='none')

        processed_frame, activityDetected = processDetection(frame, detections)

        if (activityDetected and (waitTimer is None) and (timeLeft is None)):
            waitTimer = time.time()
            timeLeft = startThreshold
            cv2.putText(processed_frame, "Device will start in {} seconds".format(timeLeft), (int(0.03 * w)+120,40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,200,0), 1)
        elif (activityDetected and timeLeft is not None and timeLeft>0):
            timeLeft = startThreshold - int(time.time() - waitTimer)
            cv2.putText(processed_frame, "Device will start in {} seconds".format(timeLeft), (int(0.03 * w)+120,40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,255), 1)
        elif (activityDetected):
            # Call to start devices
            if timeLeft is not None:
                startDevice()
            timeLeft = None
            cv2.putText(processed_frame, "Device working", (int(0.03 * w)+120,40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
        elif (not activityDetected and waitTimer is not None and timeLeft is None):
            waitTimer = time.time()
            timeLeft = stopThreshold
            cv2.putText(processed_frame, "Device will stop in {} seconds".format(timeLeft), (int(0.03 * w)+120,40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,200,0), 1)
        elif (not activityDetected and timeLeft is not None and timeLeft>0):
            timeLeft = stopThreshold - int(time.time() - waitTimer)
            cv2.putText(processed_frame, "Device will stop in {} seconds".format(timeLeft), (int(0.03 * w)+120,40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,200,255), 1)
        else:
            #Call to stop device
            if timeLeft is not None:
                stopDevice()
            waitTimer = None
            timeLeft = None
            cv2.putText(processed_frame, "Device stopped", (int(0.03 * w)+120,40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)



        fps_end = time.time()
        #print("FPS: {}".format(round(1/(fps_end-fps_start))))
        cv2.putText(processed_frame, "{}".format(round(1/(fps_end-fps_start))), (int(0.03 * w)+140, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.imshow("Detections", processed_frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
