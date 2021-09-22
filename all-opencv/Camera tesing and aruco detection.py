import cv2
from cv2 import aruco

aruco_dict = aruco.Dictionary_get(aruco.DICT_7X7_250)

vid = cv2.VideoCapture(0, cv2.CAP_DSHOW)

vid.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while(1):
    __, frame = vid.read()
    #print(frame.shape[0], frame.shape[1])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)

    frame_markers = aruco.drawDetectedMarkers(frame.copy(), corners, ids)
    
    cv2.imshow("Ids", frame_markers)
    # save tracking video to a file
    if cv2.waitKey(10) & 0xFF == ord('q'):
        vid.release()
        cv2.destroyAllWindows()
        break
