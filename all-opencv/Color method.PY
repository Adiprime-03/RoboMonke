import cv2
import numpy as np

def empty():
	pass

def getContours(img, original):
	contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	c = 0
	for cnt in contours:
		area = cv2.contourArea(cnt)
		if(area>1000):
			c = c + 1
	if(c>1):
		print("More than 1 robot of this color detected")
	else:
		for cnt in contours:
			area = cv2.contourArea(cnt)
			if(area>1000):
				peri = cv2.arcLength(cnt, True)
				approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
				x, y, w, h = cv2.boundingRect(approx)
				cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 2)

def isoriented(img):
	contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
	c = 0
	for cnt in contours:
		area = cv2.contourArea(cnt)
		if(area>1000):
			c = c + 1
	if(c>1):
		return False
	for cnt in contours:
		area = cv2.contourArea(cnt)
		if(area>1000):
			peri = cv2.arcLength(cnt, True)
			approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
			x, y, w, h = cv2.boundingRect(approx)
			narea = w*h
			if(abs(narea - area)<=2000):
				return True
			else:
				return False


'''cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 640, 240)
cv2.createTrackbar("Hue min", "Trackbars", 0, 179, empty)
cv2.createTrackbar("Hue max", "Trackbars", 179, 179, empty)
cv2.createTrackbar("Sat min", "Trackbars", 0, 255, empty)
cv2.createTrackbar("Sat max", "Trackbars", 255, 255, empty)
cv2.createTrackbar("Val min", "Trackbars", 0, 255, empty)
cv2.createTrackbar("Val max", "Trackbars", 255, 255, empty)'''


#webcam = cv2.VideoCapture(2)# or 1 depends on the USB Port being used....change accordingly



img = cv2.imread("Resources/Robots4.png")
imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


l = []

blue_lower = np.array([110, 165, 162])
blue_upper = np.array([135, 255, 255])
bl = []
bl.append(blue_lower)
bl.append(blue_upper)

red_lower = np.array([175, 98, 210])
red_upper = np.array([179, 255, 255])
re = []
re.append(red_lower)
re.append(red_upper)

yellow_lower = np.array([12, 130, 181])
yellow_upper = np.array([44, 255, 255])
ye = []
ye.append(yellow_lower)
ye.append(yellow_upper)

green_lower = np.array([44, 43, 120])
green_upper = np.array([85, 152, 255])
gr = []
gr.append(green_lower)
gr.append(green_upper)

l.append(bl)
l.append(re)
l.append(ye)
l.append(gr)



for i in range(0, 4, 1):
	lower = l[i][0]
	upper = l[i][1]
	mask = cv2.inRange(imghsv, lower, upper)

	kernel = np.ones((5, 5), np.uint8)

	imgblur = cv2.GaussianBlur(mask, (7, 7), 1)
	imgcanny = cv2.Canny(imgblur, 50, 50)
	imgdilation = cv2.cv2.dilate(imgcanny, kernel, iterations=1)

	getContours(imgdilation, img)
	print(i, " : ", isoriented(imgdilation))
	
cv2.imshow("Image", img)

	
	
'''while True:

	p = cv2.getTrackbarPos("Hue min", "Trackbars")
	u = cv2.getTrackbarPos("Hue max", "Trackbars")
	q = cv2.getTrackbarPos("Sat min", "Trackbars")
	v = cv2.getTrackbarPos("Sat max", "Trackbars")
	r = cv2.getTrackbarPos("Val min", "Trackbars")
	w = cv2.getTrackbarPos("Val max", "Trackbars")
		
	mask = cv2.inRange(imghsv, np.array([p, q, r]), np.array([u, v, w]))

	cv2.imshow("Image", img)
	cv2.imshow("Mask", mask)


	cv2.waitKey(1)'''

cv2.waitKey(0)