import numpy as np
import cv2
from cv2 import aruco
import math
import time
from matplotlib import pyplot as plt

#have to define e, x, y, t, s
#have to define start, end, fturning, sturning as tuples

e = 0
x = 0
y = 0
t = 0
s = 0

dummy = "0.0.0.0"

class robot():
    def __init__(self, id, aruco_corners, position, ip, start, fturning, sturning, end, stage=0):
        self.id = id
        self.stage = stage
        self.position = position
        self.ip = ip
        self.start = start
        self.fturning = fturning
        self.end = end
        self.aruco_corners = aruco_corners
        self.sturning = sturning

    def t1x(self):
        t1x = self.aruco_corners[0][0][0] + self.aruco_corners[0][1][0] + self.aruco_corners[0][2][0] + self.aruco_corners[0][3][0]
        t1x = t1x/8
        return t1x
    
    def t2x(self):
        t1y = self.aruco_corners[0][0][1] + self.aruco_corners[0][1][1] + self.aruco_corners[0][2][1] + self.aruco_corners[0][3][1]
        t1y = t1y/8
        return t1y

    def t2x(self):
        t2x = self.aruco_corners[1][0][0] + self.aruco_corners[1][1][0] + self.aruco_corners[1][2][0] + self.aruco_corners[1][3][0]
        t2x = t2x/8
        return t2x
    
    def t2y(self):
        t2y = self.aruco_corners[1][0][1] + self.aruco_corners[1][1][1] + self.aruco_corners[1][2][1] + self.aruco_corners[1][3][1]
        t2y = t2y/8
        return t2y


aruco_dict = aruco.Dictionary_get(aruco.DICT_7X7_250)

vid = cv2.VideoCapture(0)

# robot1


graph, (plot1) = plt.subplots(1)

plt.gca().invert_yaxis()


r = []
r.append(robot(1, [], [0, 0], dummy, [], [], [], []))
r.append(robot(2, [], [0, 0], dummy, [], [], [], []))
r.append(robot(3, [], [0, 0], dummy, [], [], [], []))
r.append(robot(4, [], [0, 0], dummy, [], [], [], []))
for i in range(0, 1):
    timelist = []
    X = []
    Y = []
    while(r[i].stage != 8):
        __, frame = vid.read()
        #print(frame.shape[0], frame.shape[1])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters)
        
        frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
        k = 0
        #         [[1]] - ids
        # [array([[[262.,  52.],
        #         [508.,  69.],
        #         [494., 325.],
        #         [221., 286.]]], dtype=float32)]
        a = 0
        b = 0
        
        if(len(corners)==1):
            for j in ids:
                if(j[0]==r[i].id):
                    a = corners[0][0][0][0] + corners[0][0][1][0] + corners[0][0][2][0] + corners[0][0][3][0]
                    b = corners[0][0][0][1] + corners[0][0][1][1] + corners[0][0][2][1] + corners[0][0][3][1]
                    a = a/4.0
                    b = b/4.0
            r[i].position.pop()
            r[i].position.pop()
            r[i].position.append(a)
            r[i].position.append(b)
            print(a, b)
            X.append(a)
            Y.append(640 - b)
            plt.xlim(0, 480)
            plt.ylim(0, 640)
            
            plt.plot(X, Y, "b")
            plt.pause(0.05)





        
            #print("fucking corner errors")
            

            #send r[i].stage to the robot

            
        
    
        cv2.imshow("Ids", frame_markers)
        #save tracking video to a file
        if cv2.waitKey(10) & 0xFF == ord('q'):
            vid.release()
            cv2.destroyAllWindows()
            break

    i = i + 1
plt.show()