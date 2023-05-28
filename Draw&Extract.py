import math
import time, cv2
import numpy as np
from tkinter import filedialog
from sklearn.mixture import GaussianMixture
from roipoly import RoiPoly
from matplotlib import pyplot as plt

my_dpi = 96

# function to check if point is inside polygon
def isInside(x, y, xp, yp):
    c = False
    j = len(xp)-1
    for i in range(len(xp)):
        if (((yp[i] > y) != (yp[j] > y)) and (x < (xp[j]-xp[i]) * (y-yp[i]) / (yp[j]-yp[i]) + xp[i])):
            c = not c
        j = i
    return c

def get_roi_coordinates(self):
    """Get co-ordinates of the ROI polygon.
    Returns
    -------
    numpy array (2D)
    """
    roi_coordinates = list(zip(self.x, self.y))
    return roi_coordinates


def main():
    path = r"C:\Users\albon\Pictures"
    file_types = [('Image', '*.jpg;*.png'), ('All files', '*')]
    name = filedialog.askopenfilename(title='Select a file', filetypes=file_types, initialdir=path)
    image = cv2.imread(name, -1)
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    IMG_SIZE = image.shape
    plt.figure(figsize=(IMG_SIZE[0]/my_dpi, IMG_SIZE[1]/my_dpi), dpi=my_dpi)
    plt.axis('off')
    plt.imshow(image1)

    # get ROI
    roi = RoiPoly(color='r')
    coords = get_roi_coordinates(roi)
    #coords = roi.get_roi_coordinates()
    xr = []             # x coordinates of ROI
    yr = []             # y coordinates of ROI
    # coords = coords[0]
    for tuple in coords:
        xr.append(tuple[0])
        yr.append(tuple[1])

    for i in range(IMG_SIZE[0]):
        for j in range(IMG_SIZE[1]):
            if not isInside(j, i, xr, yr):
                image[i,j,0] = 0
                image[i,j,1] = 0
                image[i,j,2] = 0
    
    plt.figure(figsize=(IMG_SIZE[0]/my_dpi, IMG_SIZE[1]/my_dpi), dpi=my_dpi)
    image1 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.axis('off')
    plt.imshow(image1)
    plt.show()


main()