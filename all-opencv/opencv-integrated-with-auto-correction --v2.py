import numpy as np
import cv2
from cv2 import aruco
import math
import time
from matplotlib import pyplot as plt
from intersect import intersection
import socket
from datetime import datetime as dt


# have to define e, x, y, t, s
# have to define start, end, fturning, sturning as tuples


e = 0
x = 30
y = 30
t = 100000
side_length = 0

radius = 100

p = math.pi

dummyip = '0.0.0.0'


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

    '''def t1x(self):
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
        return t2y'''


aruco_dict = aruco.Dictionary_get(aruco.DICT_7X7_250)

vid = cv2.VideoCapture(1, cv2.CAP_DSHOW)
vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter('submission.avi', fourcc, 20.0, (640, 480))

# robot1

dim = (1280, 720)

graph, (plot1) = plt.subplots(1)

plt.gca().invert_yaxis()

phi = np.arange(0, 360, 2)

port = 8090

r = []
r.append(robot(1, [0, 0], '192.168.19.230', [574, 39], [110, 475], np.array(
    [574, 574, 110]), np.array([681, 245, 245]), "R"))
r.append(robot(2, [0, 0], '192.168.19.109', [630, 35], [113, 531], np.array(
    [630, 630, 113]), np.array([685, 189, 189]), "R"))
r.append(robot(3, [0, 0], '192.168.19.172', [681, 35], [1197, 557], np.array(
    [681, 681, 1197]), np.array([685, 163, 163]), "L"))
r.append(robot(4, [0, 0], '192.168.19.26', [729, 33], [1194, 489], np.array(
    [729, 729, 1194]), np.array([687, 231, 231]), "L"))

client = []

client.append(socket.socket())
print("Waiting for connection")
try:
    client[0].connect((r[0].hostip, port))
    print("Connected!!")
except socket.error as e:
    print(str(e))

# client.append(socket.socket())
# print("Waiting for connection")
# try:
#     client[1].connect((r[1].hostip, port))
#     print("Connected!!")
# except socket.error as e:
#     print(str(e))

# client.append(socket.socket())
# print("Waiting for connection")
# try:
#     client[2].connect((r[2].hostip, port))
#     print("Connected!!")
# except socket.error as e:
#     print(str(e))

# client.append(socket.socket())
# print("Waiting for connection")
# try:
#     client[3].connect((r[3].hostip, port))
#     print("Connected!!")
# except socket.error as e:
#     print(str(e))

for i in range(0, 4):
    instant = time.time()
    flag1 = 0
    flag2 = 0
    corner = []
    corner.append(r[i].start[0])
    corner.append(720 - r[i].end[1])
    while(r[i].stage != 5):
        __, frame = vid.read()
        #print(frame.shape[0], frame.shape[1])
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters)

        frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
        k = 0

        now = dt.now()

        now = now.strftime("%m/%d/%Y, %H:%M:%S")

        #         [[1]] - ids
        # [array([[[262.,  52.],
        #         [508.,  69.],
        #         [494., 325.],
        #         [221., 286.]]], dtype=float32)]

        a = 0  # x coordinates
        b = 0  # y coordinates

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
                k = k + 1
            r[i].position.pop()
            r[i].position.pop()
            r[i].position.append(a)
            r[i].position.append(dim[1]-b)
            print("Robot is at ", r[i].position)

            # X.append(a)
            #Y.append(dim[1] - b)
            #plt.xlim(0, dim[0])
            #plt.ylim(0, dim[1])

            x2 = radius*np.cos(phi*p/180) + r[i].position[0]
            y2 = radius*np.sin(phi*p/180) + r[i].position[1]
            xin, yin = intersection(r[i].x1, r[i].y1, x2, y2)

            if(r[i].stage == 0):
                # send message to go forward
                client[i].send(str.encode("F"))
                print("Go Forward")
                print("F")
                if(abs(b-r[i].start[1]) > radius):
                    r[i].stage = 1
            elif(r[i].stage == 1):
                #print(len(xin))
                if(len(xin) == 2):
                    if(xin[0] == xin[1] or yin[1] == yin[0]):
                        client[i].send(str.encode("F"))
                    else:
                        if(flag1 == 0 and abs(corner[0]-r[i].position[0])<20 and abs(corner[1]-r[i].position[1])<20):
                            print("go ", r[i].direction)
                            client[i].send(str.encode(r[i].direction))
                            client[i].send(str.encode(r[i].direction))
                            client[i].send(str.encode(r[i].direction))
                            if(corners[0][0][0][1]==corners[0][0][1][1] and abs(corners[0][0][0][1]-corners[0][0][1][1])<side_length*1.1):
                                flag1 = 1
                            elif(corners[0][0][0][1]==corners[0][0][2][1] and abs(corners[0][0][0][1]-corners[0][0][2][1])<side_length*1.1):
                                flag1 = 1
                            elif(corners[0][0][0][1]==corners[0][0][3][1] and abs(corners[0][0][0][1]-corners[0][0][3][1])<side_length*1.1):
                                flag1 = 1
                            
                        else:
                            print("go forwards")
                            client[i].send(str.encode("F"))
                    
                        # send message
                # elif(len(xin)==1):
                #     if(r[i].direction == "right"):
                #         print("go to ", xin, yin)
                #     else:
                #         print("go to ", 2*a - xin, yin)
                # elif(len(xin)==0):
                #     if(r[i].end[0]-a>e):
                #         print("go forwards straight")
                #     else:
                #         print ("stop")
                #         r[i].stage = 2
                #         instant = time.time()
                elif(len(xin) == 1):
                    if(abs(xin - r[i].end[0]) < 75):
                        print("Stop")
                        client[i].send(str.encode("S"))
                        r[i].stage = 2
                        instant = time.time()
                    else:
                        client[i].send(str.encode("F"))
                elif(len(xin) == 3 or len(xin) == 4):
                    if(flag1 == 0):
                        print("go ", r[i].direction)
                        client[i].send(str.encode(r[i].direction))
                        if(corners[0][0][0][1]==corners[0][0][1][1] and abs(corners[0][0][0][1]-corners[0][0][1][1])<side_length*1.1):
                            flag1 = 1
                        elif(corners[0][0][0][1]==corners[0][0][2][1] and abs(corners[0][0][0][1]-corners[0][0][2][1])<side_length*1.1):
                            flag1 = 1
                        elif(corners[0][0][0][1]==corners[0][0][3][1] and abs(corners[0][0][0][1]-corners[0][0][3][1])<side_length*1.1):
                            flag1 = 1
                    elif(flag1==1):
                        print("go forwards")
                        client[i].send(str.encode("F"))
                else:
                    print("no intersection point")
                    client[i].send(str.encode("S"))
                    r[i].stage = 2
                    instant = time.time()
                #plt.plot(x1, y1, c="r")
                #plt.plot(x2, y2, c="g")
                #plt.plot(xin, yin, "*k")
                print("Intersection points - ", xin, yin)

                #plt.plot(X, Y, "b")
                # plt.pause(0.05)

            elif(r[i].stage == 2):
                time_var = time.time()
                print("run servo motor")
                client[i].send(str.encode("X"))
                # send message
                if(time_var-instant > t):
                    r[i].stage = 3
            elif(r[i].stage == 3):
                print("Go backward")
                client[i].send(str.encode("V"))
                if(abs(a-r[i].end[0]) > y):
                    r[i].stage = 4
            # elif(r[i].stage == 3):
            #     #send message to move forward
            #     if(len(xin)==0):
            #         print("go backwards straight")
            #     elif(len(xin)==1):
            #         if(yin == r[i].y1[1]):
            #             print("go to ", 2*a - xin, yin)
            #         else:
            #             if(b-r[i].start[1]>e):
            #                 print("go backwards straight")
            #             else:
            #                 print("stop")
            #                 r[i].stage = 4
            #     elif(len(xin)==2):
            #         if(xin[0]==xin[1]):
            #             if(yin[0]>yin[1]):
            #                 print("go to ", xin[0], yin[0])
            #             else:
            #                 print("go to ", xin[1], yin[1])
            #         else:
            #             print("go ", r[i].direction)#send message

            #     print("Intersection points - ", xin, yin)

            elif(r[i].stage == 4):
                if(len(xin) == 2):
                    if(xin[0] == xin[1]):
                        if(yin[0] < yin[1]):
                            print("go towards", xin[1], yin[1])
                            client[i].send(str.encode("V"))
                            # send message to robot properly
                        else:
                            print("go towards", xin[0], yin[0])
                            client[i].send(str.encode("V"))
                            # send message to robot properly
                    elif(yin[0] == yin[1]):
                        if(xin[0] < xin[1]):
                            if(r[i].direction == "right"):
                                print("go towards", xin[1], yin[1])
                                client[i].send(str.encode("V"))
                                # send message to robot properly
                            else:
                                print("go towards", xin[0], yin[0])
                                client[i].send(str.encode("V"))
                                # send message to robot properly
                        else:
                            if(r[i].direction == "right"):
                                print("go towards", xin[0], yin[0])
                                client[i].send(str.encode("V"))
                                # send message to robot properly
                            else:
                                print("go towards", xin[1], yin[1])
                                client[i].send(str.encode("V"))
                                # send message to robot properly
                    else:
                        if(flag1 == 0):
                            print("go ", r[i].direction)
                            client[i].send(str.encode(r[i].direction))
                            if(corners[0][0][0][1]==corners[0][0][1][1] and abs(corners[0][0][0][1]-corners[0][0][1][1])<side_length*1.1):
                                flag1 = 1
                            elif(corners[0][0][0][1]==corners[0][0][2][1] and abs(corners[0][0][0][1]-corners[0][0][2][1])<side_length*1.1):
                                flag1 = 1
                            elif(corners[0][0][0][1]==corners[0][0][3][1] and abs(corners[0][0][0][1]-corners[0][0][3][1])<side_length*1.1):
                                flag1 = 1
                            
                        elif(flag1==1):
                            print("go backwars")
                            client[i].send(str.encode("V"))
                elif(len(xin == 1)):
                    if(flag1 == 0):
                        print("go ", r[i].direction)
                        client[i].send(str.encode(r[i].direction))
                        if(corners[0][0][0][1]==corners[0][0][1][1] and abs(corners[0][0][0][1]-corners[0][0][1][1])<side_length*1.1):
                            flag1 = 1
                        elif(corners[0][0][0][1]==corners[0][0][2][1] and abs(corners[0][0][0][1]-corners[0][0][2][1])<side_length*1.1):
                            flag1 = 1
                        elif(corners[0][0][0][1]==corners[0][0][3][1] and abs(corners[0][0][0][1]-corners[0][0][3][1])<side_length*1.1):
                            flag1 = 1
                            
                    elif(flag1==1):
                        print("go backwardss")
                        client[i].send(str.encode("V"))
                    # send message
                    r[i].stage = 5
                elif(len(xin) == 3 or len(xin) == 4):
                    print("go ", r[i].direction)
                    client[i].send(str.encode(r[i].direction))
                else:
                    print("Robot has gone rogue")
            #     plt.plot(r[i].x1, r[i].y1, c="r")
            #     plt.plot(x2, y2, c="g")
            #     plt.plot(xin, yin, "*k")
                print("Intersection points - ", xin, yin)

            #     #plt.plot(X, Y, "b")
            #     #plt.pause(0.05)

        elif(len(corners) == 0):
            print("show atleast 1 u dumbfuck")
            client[i].send(str.encode("S"))
        elif(len(corners) < 4):
            print(
                "Some robot is missing. Arrange properly and then run this code u fool.")
            client[i].send(str.encode("S"))
        else:
            print("How the fuck is there more than 4 robots")
            client[i].send(str.encode("S"))

        cv2.imshow("Ids", frame_markers)
        #cv2.imwrite('videogrid.avi', frame_markers)
        # save tracking video to a file
        out.write(frame_markers)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
    client[i].close()
    i = i + 1
    print("Closing connection")

vid.release()
out.release()
# plt.show()
