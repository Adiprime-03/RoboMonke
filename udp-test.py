import socket
import cv2
from cv2 import aruco

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
vid = cv2.VideoCapture(0)
aruco_dict = aruco.Dictionary_get(aruco.DICT_7X7_250)

#client = socket.socket()
#client1 = socket.socket()
host = '192.168.43.18'
port = 8090
sock.bind(host, port)
# print("Waiting for connection")
# try:
#     client.connect((host, port))
#     print("Connected!!")
# except socket.error as e:
#     print(str(e))
# try:
#     client1.connect((host, port))
#     print("Connected!!")
# except socket.error as e:
#     print(str(e))
while True:
    __, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)
    frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    cv2.imshow("Ids", frame_markers)
    data, addr = sock.recvfrom(4096)
    if len(corners) == 4:
        message = "F".encode('utf-8')
        sock.sendto(message, addr)
    else:
        message = "S".encode('utf-8')
        sock.sendto(message, addr)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        vid.release()
        cv2.destroyAllWindows()
        break
