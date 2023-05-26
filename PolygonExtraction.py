import numpy as np
import matplotlib as plt
import os
import cv2

path = r"C:\Users\albon\Desktop\Semantic Segmentation 2.0\train-masks\train\20.jpg"
img = cv2.imread(path, -1)
canvas = np.array(img)
IMG_SIZE = [canvas.shape[0], canvas.shape[1]]
polygon_pnts = []

for i in range(IMG_SIZE[0]):
    for j in range(IMG_SIZE[1]):
        if img[i,j,0] == 255 and img[i,j,1] == 255 and img[i,j,2] == 255:
            polygon_pnts.append((i,j))
        canvas[j,i,0] = 0
        canvas[j,i,1] = 0
        canvas[j,i,2] = 0  

#cv2.imshow("Polygon Extraction", img)
#cv2.waitKey()
#cv2.destroyWindow("Polygon Extraction")

for i in polygon_pnts:
    canvas[i[0],i[1],0] = 255
    canvas[i[0],i[1],1] = 255
    canvas[i[0],i[1],2] = 255  

cv2.imshow("Mask", img)
cv2.waitKey()
cv2.imshow("Polygon Extraction", canvas)
cv2.waitKey()
cv2.destroyAllWindows()