import socket
import cv2
from cv2 import aruco

vid = cv2.VideoCapture(0)
aruco_dict = aruco.Dictionary_get(aruco.DICT_7X7_250)

client = socket.socket()
host = '192.168.43.18'
port = 8090
print("Waiting for connection")
try:
    client.connect((host, port))
    print("Connected!!")
except socket.error as e:
    print(str(e))
while True:
    __, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)
    frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    cv2.imshow("Ids", frame_markers)
    if len(corners) == 1:
        client.send(str.encode("X"))
        client.send(str.encode("F"))
    else:
        client.send(str.encode("S"))
    if cv2.waitKey(10) & 0xFF == ord('q'):
        vid.release()
        cv2.destroyAllWindows()
        break
client.close()
