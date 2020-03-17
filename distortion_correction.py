import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

def undistorted(image,objpoints,imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image.shape[1::-1],None,None)
    dst = cv2.undistort(image, mtx, dist, None,mtx)
    return dst

# Make a list of calibration images
images = glob.glob('C:/test/camera_caliberation/calibration*.jpg')
objpoints = [] #they are the 3D points which represent the corners of an undistorted chessboard image
imgpoints = [] # they are the 2D points of the chessboard corners on the distorted image
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2) #mgrid function returns the co-ordinate value for a given grid size

# Step through the list and search for chessboard corners
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    '''plt.imshow(img)
    plt.show()'''
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6),None)

    # If found, add object points, image points
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9,6), corners, ret)
image = cv2.imread('C:/test/camera_caliberation/calibration1.jpg')
plt.title("distorted image")
plt.imshow(image)
plt.show()
undistorted_image = undistorted(image,objpoints,imgpoints)
plt.title("undistorted image")
plt.imshow(undistorted_image)
plt.show()
