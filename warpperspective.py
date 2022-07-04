import cv2
import numpy as np
import pandas as pd
img = cv2.imread('arena.jpg')
width = 1280
height = 720
table = pd.read_csv('warp-coords.csv')
pts1 = np.float32([[table['x'][0], table['y'][0]], [table['x'][1], table['y'][1]], [
                  table['x'][2], table['y'][2]], [table['x'][3], table['y'][3]]])
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
output = cv2.warpPerspective(img, matrix, (width, height))
cv2.imshow('original', img)
cv2.imshow('output', output)
cv2.imwrite('warped-arena.jpg', output)
cv2.waitKey(0)
