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


e = 0.3
x = 30
y = 30
t = 100000
side_length = 0

radius = 100

p = math.pi

def staright_or_gay(rvec):
    r = cv2.Rodrigues(rvec)
    v = []
    v.append(r[0][0][0])
    v.append(r[0][1][0])
    if(-1<r[0][2][2]<-0.9):
        if(0.93<v[0]<=1 and 0<=v[1]<0.07):
            return "facing straight"
        elif(-0.07<v[0]<0.07 and -1<=v[1]<-0.93):
            return "facing right"
        elif(-0.07<v[0]<0.07 and 0.93<v[1]<=1):
            return "facing left"
        elif(-1<v[0]<-0.93 and -0.07<v[1]<0.07):
            return "facing down"
        elif(0.07<=v[0]<=0.93 and 0.07<=v[1]<=0.93):
            return "gay 1"
        elif(-0.93<=v[0]<=-0.07 and 0.07<=v[1]<=0.93):
            return "gay 2"
        elif(-0.93<=v[0]<=-0.07 and -0.93<=v[1]<=-0.07):
            return "gay 3"
        elif(0.07<=v[0]<=0.93 and -0.93<=v[1]<=-0.07):
            return "gay 4"
        else:
            return "weird shit"
               
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
r.append(robot(1, [0, 0], '192.168.19.230', [table['x'][0], table['y'][0]], [table['x'][1], table['y'][1]], np.array([table['x'][0], table['x'][0], table['x'][1]]), np.array([dim[1] - table['y'][0], dim[1] - table['y'][1], dim[1] - table['y'][1]]), "right"))
r.append(robot(2, [0, 0], '192.168.19.109', [table['x'][2], table['y'][2]], [table['x'][3], table['y'][3]], np.array([table['x'][2], table['x'][2], table['x'][3]]), np.array([dim[1] - table['y'][2], dim[1] - table['y'][3], dim[1] - table['y'][3]]), "right"))
r.append(robot(3, [0, 0], '192.168.19.172', [table['x'][4], table['y'][4]], [table['x'][5], table['y'][5]], np.array([table['x'][4], table['x'][4], table['x'][5]]), np.array([dim[1] - table['y'][4], dim[1] - table['y'][5], dim[1] - table['y'][5]]), "left"))
r.append(robot(4, [0, 0], '192.168.19.26', [table['x'][6], table['y'][6]], [table['x'][7], table['y'][7]], np.array([table['x'][6], table['x'][6], table['x'][7]]), np.array([dim[1] - table['y'][6], dim[1] - table['y'][7], dim[1] - table['y'][7]]), "left"))

for i in range(0, 4, 1):
    corner = []
    corner.append(r[i].start[0])
    corner.append(dim[1] - r[i].end[1])
    while(r[i].stage is not 8):
        
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


            x2 = radius*np.cos(phi*p/180) + r[i].position[0]
            y2 = radius*np.sin(phi*p/180) + r[i].position[1]
            xin, yin = intersection(r[i].x1, r[i].y1, x2, y2)

            print(check)

            if(r[i].stage == 0):
                if(abs(b-r[i].start[1]) > radius):
                    r[i].stage = 1
                else:
                    if(check == "facing straight"):
                        print("F")
                    elif(check == "gay 1"):
                        print("R")
                    elif(check == "gay 4"):
                        print("L")
                    else:
                        print("Robot gone rogue")
                        print("S")
                    
            elif(r[i].stage == 1):
                if(abs(r[i].position[1]-corner[1])<e):
                    print("S")
                    r[i].stage = 2
                else:
                    if(check == "facing straight"):
                        print("F")
                    elif(check == "gay 1"):
                        print("R")
                    elif(check == "gay 4"):
                        print("L")
                    else:
                        print("Robot gone rogue")
                        print("S")
            
            elif(r[i].stage == 2):
                if(check == "facing " + r[i].direction):
                    r[i].stage = 3
                else:
                    if(r[i].direction == "right"):
                        print("R")
                    else:
                        print("L")
            
            elif(r[i].stage == 3):
                if(abs(r[i].position[0]-r[i].end[0])<e):
                    print("S")
                    r[i].stage = 4
                    instant = time.time()
                else:
                    if(r[i].direction == "right"):
                        if(check == "facing " + r[i].direction):
                            print("F")
                        elif(check == "gay 3"):
                            print("L")
                        elif(check == "gay 4"):
                            print("R")
                        else:
                            print("Robot gone rogue")
                            print("S")
                    else:
                        if(check == "facing " + r[i].direction):
                            print("F")
                        elif(check == "gay 1"):
                            print("L")
                        elif(check == "gay 2"):
                            print("R")
                        else:
                            print("Robot gone rogue")
                            print("S")
            
            elif(r[i].stage == 4):
                print("X")
                tau = time.time()
                if(tau - instant >t):
                    r[i].stage = 5
            
            elif(r[i].stage == 5):
                if(abs(r[i].position[0]-corner[0])<e):
                    print("S")
                    r[i].stage = 6
                else:
                    if(r[i].direction == "right"):
                        if(check == "facing " + r[i].direction):
                            print("V")
                        elif(check == "gay 3"):
                            print("L")
                        elif(check == "gay 4"):
                            print("R")
                        else:
                            print("Robot gone rogue")
                            print("S")
                    else:
                        if(check == "facing " + r[i].direction):
                            print("V")
                        elif(check == "gay 1"):
                            print("L")
                        elif(check == "gay 2"):
                            print("R")
                        else:
                            print("Robot gone rogue")
                            print("S")

            elif(r[i].stage == 6):
                if(check == "facing " + r[i].direction):
                    r[i].stage = 7
                else:
                    if(r[i].direction == "right"):
                        print("R")
                    else:
                        print("L")
            
            elif(r[i].stage == 7):
                if(abs(r[i].position[1]-r[i].start[1])<e):
                        print("S")
                        r[i].stage = 8
                else:
                    if(check == "facing straight"):
                        print("V")
                    elif(check == "gay 1"):
                        print("R")
                    elif(check == "gay 4"):
                        print("L")
                    else:
                        print("Robot gone rogue")
                        print("S")
        
            else:
                print("Unknown stage")
                print("S")
        
        elif(0<=len(corners)<4):
            print("Few robots are missing")
        else:
            print("idk what's happening")
        
        cv2.imshow("Ids", frame)
        #cv2.imwrite('videogrid.avi', frame_markers)
        # save tracking video to a file
        out.write(frame)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    i = i + 1
vid.release()
out.release()