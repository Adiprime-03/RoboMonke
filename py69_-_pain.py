import numpy as np
import cv2
from cv2 import Rodrigues, aruco
import math
import time
from matplotlib import pyplot as plt
from intersect import intersection
import socket
from datetime import datetime
import csv
import pandas as pd
import Calibration
# binding all IP
HOST = '0.0.0.0'
# listening on port
PORT = 44444
# size of receive buffer
BUFFER_SIZE = 1024
# Creating UDP socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.bind((HOST, PORT))

# have to define e, x, y, t, s
# have to define start, end, fturning, sturning as tuples

e = 25
x = 30
y = 30
t = 2
side_length = 0

radius = 100

p = math.pi


Font = cv2.FONT_HERSHEY_COMPLEX


def angle_gay(index, corners):
    theta = np.arctan((corners[index][0][1][1]-corners[index][0]
                      [0][1])/(corners[index][0][1][0]-corners[index][0][0][0]))

    theta = np.degrees(theta)

    if(abs(theta) < 10):
        if(corners[index][0][1][0] > corners[index][0][0][0]):
            return "straight", 0
        else:
            return "back", 180
    elif(80 < abs(theta) < 90):
        if(corners[index][0][1][1] > corners[index][0][0][1]):
            return "right", 90
        else:
            return "left", 270
    elif(theta > 0):
        if(corners[index][0][1][0] > corners[index][0][0][0]):
            return "gay 4", theta
        else:
            return "gay 2", 180 + theta
    elif(theta < 0):
        if(corners[index][0][1][0] > corners[index][0][0][0]):
            return "gay 1", 360 + theta
        else:
            return "gay 3", 180 + theta

    else:
        return "nothing", 8


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
width = 1280
height = 720
table1 = pd.read_csv('warp-coords.csv')
pts1 = np.float32([[table1['x'][0], table1['y'][0]], [table1['x'][1], table1['y'][1]], [
                  table1['x'][2], table1['y'][2]], [table1['x'][3], table1['y'][3]]])
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter('lastday1.avi', fourcc, 20.0, (1280, 720))

# robot1

dim = (1280, 720)

graph, (plot1) = plt.subplots(1)

plt.gca().invert_yaxis()

phi = np.arange(0, 360, 2)

table = pd.read_csv('coords.csv')

r = []
r.append(robot(1, [0, 0], '192.168.137.123', [table['x'][0], table['y'][0]], [table['x'][1], table['y'][1]], np.array(
    [table['x'][0], table['x'][0], table['x'][1]]), np.array([dim[1] - table['y'][0], dim[1] - table['y'][1], dim[1] - table['y'][1]]), "right"))
r.append(robot(2, [0, 0], '192.168.137.11', [table['x'][2], table['y'][2]], [table['x'][3], table['y'][3]], np.array(
    [table['x'][2], table['x'][2], table['x'][3]]), np.array([dim[1] - table['y'][2], dim[1] - table['y'][3], dim[1] - table['y'][3]]), "right"))
r.append(robot(3, [0, 0], '192.168.137.243', [table['x'][4], table['y'][4]], [table['x'][5], table['y'][5]], np.array(
    [table['x'][4], table['x'][4], table['x'][5]]), np.array([dim[1] - table['y'][4], dim[1] - table['y'][5], dim[1] - table['y'][5]]), "left"))
r.append(robot(4, [0, 0], '192.168.137.237', [table['x'][6], table['y'][6]], [table['x'][7], table['y'][7]], np.array(
    [table['x'][6], table['x'][6], table['x'][7]]), np.array([dim[1] - table['y'][6], dim[1] - table['y'][7], dim[1] - table['y'][7]]), "left"))


for i in range(0, 4, 1):
    corner = []
    corner.append(r[i].start[0])
    corner.append(dim[1] - r[i].end[1])
    instant = time.time()
    servo_flag = 0
    while(r[i].stage != 8):

        __, frame = vid.read()
        frame = cv2.warpPerspective(frame, matrix, (width, height))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters)

        aruco.drawDetectedMarkers(frame, corners, ids)

        index = 0

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
                    index = k
                k = k + 1

            r[i].position.pop()
            r[i].position.pop()
            r[i].position.append(a)
            r[i].position.append(dim[1]-b)
            print("Robot is at ", r[i].position)

            x2 = radius*np.cos(phi*p/180) + r[i].position[0]
            y2 = radius*np.sin(phi*p/180) + r[i].position[1]
            xin, yin = intersection(r[i].x1, r[i].y1, x2, y2)

            check, theta = angle_gay(index, corners)
            print(check)

            if(r[i].stage == 0):
                if(len(xin) == 2):
                    r[i].stage = 1
                else:
                    print("F")
                    s.sendto("F".encode(), (r[i].hostip, PORT))

            elif(r[i].stage == 1):
                if((r[i].position[1]-corner[1]) < 25):
                    print("S")
                    s.sendto("S".encode(), (r[i].hostip, PORT))
                    r[i].stage = 2
                else:
                    if(check == "gay 1"):

                        print("r")
                        s.sendto("r".encode(), (r[i].hostip, PORT))
                    elif(check == "gay 4"):
                        print("l")
                        s.sendto("l".encode(), (r[i].hostip, PORT))
                    elif(check == "straight"):
                        if(r[i].id == 2):
                            s.sendto("r".encode(), (r[i].hostip, PORT))
                        elif(r[i].id == 3):
                            s.sendto("L".encode(), (r[i].hostip, PORT))
                        print("F")
                        s.sendto("F".encode(), (r[i].hostip, PORT))
                    else:
                        print("R")
                        s.sendto("R".encode(), (r[i].hostip, PORT))

            elif(r[i].stage == 2):
                if(check == r[i].direction):
                    print("S")
                    s.sendto("S".encode(), (r[i].hostip, PORT))
                    r[i].stage = 3
                else:
                    if(r[i].direction == "right"):
                        print("R")
                        s.sendto("R".encode(), (r[i].hostip, PORT))
                    else:
                        print("L")
                        s.sendto("l".encode(), (r[i].hostip, PORT))

            elif(r[i].stage == 3):
                if(abs(r[i].position[0]-r[i].end[0]) < 5):
                    print("S")
                    s.sendto("S".encode(), (r[i].hostip, PORT))
                    instant = time.time()
                    r[i].stage = 4
                else:
                    if(check == r[i].direction):
                        print("F")
                        s.sendto("F".encode(), (r[i].hostip, PORT))
                    elif(check == "gay 3" or check == "gay 1"):
                        print("l")
                        s.sendto("l".encode(), (r[i].hostip, PORT))
                    elif(check == "gay 2" or check == "gay 4"):
                        print("r")
                        s.sendto("r".encode(), (r[i].hostip, PORT))
                    else:
                        print("R")
                        s.sendto("R".encode(), (r[i].hostip, PORT))

            elif(r[i].stage == 4):
                tau = time.time()
                if(tau - instant > t):
                    print("S")
                    s.sendto("S".encode(), (r[i].hostip, PORT))
                    r[i].stage = 5
                elif(servo_flag == 0):
                    servo_flag = 1
                    print("X")
                    s.sendto("X".encode(), (r[i].hostip, PORT))
                else:
                    continue

            elif(r[i].stage == 5):
                if((r[i].id == 2 or r[i].id == 3) and abs(r[i].position[0]-corner[0]) < 50):
                    print("S")
                    s.sendto("S".encode(), (r[i].hostip, PORT))
                    r[i].stage = 6
                elif(abs(r[i].position[0]-corner[0]) < 30):
                    print("S")
                    s.sendto("S".encode(), (r[i].hostip, PORT))
                    r[i].stage = 6
                else:
                    if(check == r[i].direction):

                        print("V")
                        s.sendto("V".encode(), (r[i].hostip, PORT))
                    elif(check == "gay 3" or check == "gay 1"):
                        print("l")
                        s.sendto("l".encode(), (r[i].hostip, PORT))
                    elif(check == "gay 2" or check == "gay 4"):
                        print("r")
                        s.sendto("r".encode(), (r[i].hostip, PORT))
                    else:
                        print("R")
                        s.sendto("R".encode(), (r[i].hostip, PORT))

            elif(r[i].stage == 6):
                if(check == "straight"):
                    print("S")
                    s.sendto("S".encode(), (r[i].hostip, PORT))
                    r[i].stage = 7
                else:
                    if(r[i].direction == "right"):

                        print("L")
                        if(r[i].id == 1):
                            s.sendto("l".encode(), (r[i].hostip, PORT))
                        else:

                            s.sendto("l".encode(), (r[i].hostip, PORT))
                    else:
                        print("R")
                        s.sendto("r".encode(), (r[i].hostip, PORT))

            elif(r[i].stage == 7):
                if(abs(r[i].position[1]-(dim[1]-r[i].start[1])) < 10):
                    print("S")
                    s.sendto("S".encode(), (r[i].hostip, PORT))
                    r[i].stage = 8
                else:
                    if(check == "straight"):
                        print("V")
                        s.sendto("V".encode(), (r[i].hostip, PORT))
                    elif(check == "gay 1"):
                        print("r")
                        s.sendto("r".encode(), (r[i].hostip, PORT))
                    elif(check == "gay 4"):
                        print("l")
                        s.sendto("l".encode(), (r[i].hostip, PORT))
                    else:
                        print("Robot gone rogue")
                        print("S")
                        s.sendto("S".encode(), (r[i].hostip, PORT))

            else:
                print("Unknown stage")
                print("S")
                s.sendto("S".encode(), (r[i].hostip, PORT))

        elif(0 <= len(corners) < 4):
            print("Few robots are missing")

        else:
            print("idk what's happening")

        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        cv2.putText(frame, dt_string, (500, 40), Font, 2, (0, 0, 255), 2)
        cv2.imshow("Ids", frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    i = i + 1
s.close()
vid.release()
out.release()
