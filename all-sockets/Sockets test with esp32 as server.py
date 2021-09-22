import socket
import cv2
from cv2 import aruco

vid = cv2.VideoCapture(0)
aruco_dict = aruco.Dictionary_get(aruco.DICT_7X7_250)

client = []

client.append(socket.socket())
client.append(socket.socket())
host = '192.168.43.18'
host1 = ''
port = 8090
print("Waiting for connection")
try:
    client[0].connect((host, port))
    print("Connected!!")
except socket.error as e:
    print(str(e))
print("Waiting for connection")
try:
    client[1].connect((host1, port))
    print("Connected!!")
except socket.error as e:
    print(str(e))
for i in range(0, 2):
    while True:
        __, frame = vid.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        parameters = aruco.DetectorParameters_create()
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
            gray, aruco_dict, parameters=parameters)
        frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
        cv2.imshow("Ids", frame_markers)
        if ids == 2:
            client[i].send(str.encode("F"))
        else:
            client[i].send(str.encode("S"))
        if cv2.waitKey(10) & 0xFF == ord('q'):
            vid.release()
            cv2.destroyAllWindows()
            break
    client[i].close()
    i = i+1