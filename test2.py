import socket
import numpy as np
import cv2
from cv2 import aruco
s = socket.socket()

s.bind(('0.0.0.0', 8090))
s.listen(0)

vid = cv2.VideoCapture(1)
aruco_dict = aruco.Dictionary_get(aruco.DICT_7X7_250)
while True:

    client, addr = s.accept()
    print(client)
    while True:
        __, frame = vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters)
        frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
        cv2.imshow("Ids", frame_markers)
        # if ids == 1:
        client.send("F".encode())
        if cv2.waitKey(10) & 0xFF == ord('q'):
            vid.release()
            cv2.destroyAllWindows()
            break
    print("Closing connection")
    client.close()
    # client1.close()
