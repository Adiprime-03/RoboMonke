import Calibration
import cv2
import numpy as np
import time
from cv2 import aruco



aruco_dict = aruco.Dictionary_get(aruco.DICT_7X7_50)

matrix_coefficients, distortion_coefficients = Calibration.calibration()

vid = cv2.VideoCapture(0)

fourcc = cv2.VideoWriter_fourcc(*'XVID')

out = cv2.VideoWriter('submission.avi', fourcc, 20.0, (640, 480))

Font = cv2.FONT_HERSHEY_COMPLEX

while(1):
    __, frame = vid.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejectedImgPoints = aruco.detectMarkers(
        gray, aruco_dict, parameters=parameters)

    try:
        if np.all(ids is not None):  # If there are markers found by detector
            for i in range(0, len(ids)):  # Iterate in markers
                # Estimate pose of each marker and return the values rvec and tvec---different from camera coefficients
                rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                           distortion_coefficients)
                (rvec - tvec).any()  # get rid of that nasty numpy value array error
                # Draw A square around the markers
                aruco.drawDetectedMarkers(frame, corners)
                aruco.drawAxis(frame, matrix_coefficients,
                               distortion_coefficients, rvec, tvec, 0.01)  # Draw axis

                c_x = (corners[i][0][0][0] + corners[i][0][1][0] + corners[i][0][2]
                       [0] + corners[i][0][3][0]) / 4  # X coordinate of marker's center
                c_y = (corners[i][0][0][1] + corners[i][0][1][1] + corners[i][0][2]
                       [1] + corners[i][0][3][1]) / 4  # Y coordinate of marker's center
                cv2.putText(frame, "id"+str(ids[i]), (int(c_x), int(c_y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 225, 250), 2)
                
                #print(f'{time.time()}  :  {rvec}, {tvec}')
                s = tvec[0][0][0]
                g = str(rvec[0][0][0])
                r = cv2.Rodrigues(rvec)
                print(r[0])
                if(-1<r[0][2][2]<-0.9):
                    if((0.95<abs(r[0][0][0])<=1 and 0.95<abs(r[0][1][1])<=1) or  (0.95<abs(r[0][0][1])<=1 and 0.95<abs(r[0][1][0])<=1)):
                        cv2.putText(frame, "straight", (10, 460), Font, 2, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "gay", (10, 460), Font, 2, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "not oriented properly", (10, 460), Font, 2, (0, 0, 255), 2)
                
    except:
        if ids is None or len(ids) == 0:
            print("**")
            print("*** Marker Detection Failed **")
            print("**")

    out.write(frame)
    cv2.imshow("image", frame)
    time.sleep(0)
    # save tracking video to a file
    if cv2.waitKey(10) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

out.release()
vid.release()