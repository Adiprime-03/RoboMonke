import numpy as np
import cv2
from cv2 import Rodrigues, aruco
import math
import time
from matplotlib import pyplot as plt
from intersect import intersection
import socket
from datetime import datetime as dt
import csv
import pandas as pd
import Calibration


# have to define e, x, y, t, s
# have to define start, end, fturning, sturning as tuples


e = 0
x = 30
y = 30
t = 100000
side_length = 0

radius = 100

p = math.pi

def staright_or_gay(rvec):
    r = cv2.Rodrigues(rvec)
    if(-1<r[0][2][2]<-0.9):
        if((0.95<abs(r[0][0][0])<=1 and 0.95<abs(r[0][1][1])<=1) or  (0.95<abs(r[0][0][1])<=1 and 0.95<abs(r[0][1][0])<=1)):
            return "straight"
        else:
            return "gay"
    else:
        return "not oriented"


class robot():
    def __init__(self, id, position, hostip, start, end, x1, y1, direction, stage=0):
        self.id = id
        self.stage = stage
        self.position = position
        self.hostip = hostip
        self.start = start
        self.end = end
        self.direction = direction
        self.x1 = x1
        self.y1 = y1

aruco_dict = aruco.Dictionary_get(aruco.DICT_7X7_250)

vid = cv2.VideoCapture(1, cv2.CAP_DSHOW)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
matrix_coefficients, distortion_coefficients = Calibration.calibration()

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter('submission.avi', fourcc, 20.0, (1280, 720))

# robot1

dim = (1280, 720)

graph, (plot1) = plt.subplots(1)

plt.gca().invert_yaxis()

phi = np.arange(0, 360, 2)

table = pd.read_csv('coords.csv')

r = []
r.append(robot(1, [0, 0], '192.168.19.230', [table['x'][0], table['y'][0]], [table['x'][1], table['y'][1]], np.array([table['x'][0], table['x'][0], table['x'][1]]), np.array([dim[1] - table['y'][0], dim[1] - table['y'][1], dim[1] - table['y'][1]]), "R"))
r.append(robot(2, [0, 0], '192.168.19.109', [table['x'][2], table['y'][2]], [table['x'][3], table['y'][3]], np.array([table['x'][2], table['x'][2], table['x'][3]]), np.array([dim[1] - table['y'][2], dim[1] - table['y'][3], dim[1] - table['y'][3]]), "R"))
r.append(robot(3, [0, 0], '192.168.19.172', [table['x'][4], table['y'][4]], [table['x'][5], table['y'][5]], np.array([table['x'][4], table['x'][4], table['x'][5]]), np.array([dim[1] - table['y'][4], dim[1] - table['y'][5], dim[1] - table['y'][5]]), "L"))
r.append(robot(4, [0, 0], '192.168.19.26', [table['x'][6], table['y'][6]], [table['x'][7], table['y'][7]], np.array([table['x'][6], table['x'][6], table['x'][7]]), np.array([dim[1] - table['y'][6], dim[1] - table['y'][7], dim[1] - table['y'][7]]), "L"))

for i in range(0, 4, 1):
    while(True):

        __, frame = vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        aruco.drawDetectedMarkers(frame, corners, ids)

        check = ""

        if(len(corners) == 4):
            k = 0
            for j in ids:
                if(j[0] == r[i].id):
                    a = corners[k][0][0][0] + corners[k][0][1][0] + \
                        corners[k][0][2][0] + corners[k][0][3][0]
                    b = corners[k][0][0][1] + corners[k][0][1][1] + \
                        corners[k][0][2][1] + corners[k][0][3][1]
                    a = a/4.0
                    b = b/4.0
                    rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[k], 0.02, matrix_coefficients, distortion_coefficients)
                    aruco.drawAxis(frame, matrix_coefficients, distortion_coefficients, rvec, tvec, 0.01)
                    check = check + staright_or_gay(rvec)
                k = k + 1
            r[i].position.pop()
            r[i].position.pop()
            r[i].position.append(a)
            r[i].position.append(dim[1]-b)
            print("Robot is at ", r[i].position)
