import cv2 as cv

vid = cv.VideoCapture(0)

while(1):
    __, frame = vid.read()
    cv.imshow("Ids", frame)
    cv.imwrite("cali5.png", frame)
    # save tracking video to a file
    if cv.waitKey(10) & 0xFF == ord('q'):
        vid.release()
        cv.destroyAllWindows()
        break