import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import math
import cv2
import glob
import pickle

def abs_sobel_threshold(image, orient, sobel_kernel, thresh):
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if (orient == 'x'):
        sobel = cv2.Sobel(gray_scale,cv2.CV_64F,1,0,ksize)
    else:
        sobel = cv2.Sobel(gray_scale,cv2.CV_64F,0,1,ksize)
    abs_sobel = np.absolute(sobel)
    new_abs_sobel = np.uint8(255 * (abs_sobel/np.max(abs_sobel)))
    binary_sobel_thresholding = np.zeros_like(abs_sobel)
    binary_sobel_thresholding[(new_abs_sobel >= thresh[0]) & (new_abs_sobel <= thresh[1])] = 1
    return binary_sobel_thresholding

def magnitude_threshold(image,sobel_kernel,mag_thresh):
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_scale,cv2.CV_64F,1,0,ksize)
    sobel_y = cv2.Sobel(gray_scale,cv2.CV_64F,0,1,ksize)
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    abs_sobel = np.sqrt((abs_sobel_x**2) + (abs_sobel_y**2))
    new_abs_sobel = np.uint8(255 * (abs_sobel/np.max(abs_sobel)))
    binary_sobel_thresholding = np.zeros_like(abs_sobel)
    binary_sobel_thresholding[(new_abs_sobel >= mag_thresh[0])&(new_abs_sobel <= mag_thresh[1])] = 1
    return binary_sobel_thresholding

def direction_threshold(image,sobel_kernel,thresh):
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_scale,cv2.CV_64F,1,0,ksize)
    sobel_y = cv2.Sobel(gray_scale,cv2.CV_64F,0,1,ksize)
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    abs_sobel = np.arctan2(abs_sobel_y,abs_sobel_x)
    new_abs_sobel = np.uint8(255 * (abs_sobel/np.max(abs_sobel)))
    binary_sobel_thresholding = np.zeros_like(abs_sobel)
    binary_sobel_thresholding[(new_abs_sobel >= thresh[0])&(new_abs_sobel <= thresh[1])] = 1
    return binary_sobel_thresholding

def combination(image,ksize,thresh):
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_scale,cv2.CV_64F,1,0,ksize)
    abs_sobel_x = np.absolute(sobel_x)
    new_abs_sobel_x = np.uint8(255 * (abs_sobel_x/np.max(abs_sobel_x))) #this brings the values in the range of (0,255)
    binary_sobel_thresholding = np.zeros_like(sobel_x)
    binary_sobel_thresholding[(new_abs_sobel_x >= 15)&(new_abs_sobel_x <= thresh[1])] = 1
    #plt.imshow(binary_sobel_thresholding,cmap='gray')
    #plt.show()
    new_image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = new_image_hls[:,:,2]
    #plt.imshow(s_channel,cmap='gray')
    #plt.show()
    abs_s_channel = np.absolute(s_channel)
    abs_s_channel_ranging = 255*(abs_s_channel/np.max(abs_s_channel))
    binary_s_channel_thresholding = np.zeros_like(s_channel)
    binary_s_channel_thresholding[(abs_s_channel_ranging >= 170)&(abs_s_channel_ranging<=thresh[1])] = 1
    #plt.imshow(binary_s_channel_thresholding,cmap='gray')
    #plt.show()
    new_one = np.zeros_like(binary_sobel_thresholding)
    color_image = np.dstack((new_one,binary_sobel_thresholding,binary_s_channel_thresholding))
    combined_image = np.zeros_like(binary_sobel_thresholding)
    combined_image[(binary_sobel_thresholding == 1)|(binary_s_channel_thresholding == 1)] = 1
    return color_image, combined_image
#reading the image
image = cv2.imread('C:/test/bridge_shadow.jpg')
plt.title("original image")
plt.imshow(image)
plt.show()
thresh = (15,255)
ksize = 5
image_1 = abs_sobel_threshold(image,'x',ksize,thresh)
plt.title("binary image after performing sobel X operator")
plt.imshow(image_1,cmap='gray')
plt.show()
image_2 = magnitude_threshold(image,ksize,thresh)
plt.title("binary image after performing magnitude thresholding")
plt.imshow(image_2,cmap='gray')
plt.show()
image_3 = direction_threshold(image,ksize,thresh)
plt.title("binary image after performing direction thresholding")
plt.imshow(image_3,cmap='gray')
plt.show()
new_image_1,new_image_2 = combination(image,ksize,thresh)
plt.title("image with binary sobel thresholding and binary s channel thresholding")
plt.imshow(new_image_1)
plt.show()
plt.title("image with combination of both (sobel thresholding and s channel thresholding)")
plt.imshow(new_image_2,cmap='gray')
plt.show()
'''
#reading the image
image = cv2.imread('C:/test/sobel.png')
#Choose a Sobel kernel size
ksize = 5 # Choose a larger odd number to smooth gradient measurements

# Apply each of the thresholding functions
gradx = abs_sobel_threshold(image, orient='x', sobel_kernel=ksize, thresh=(20, 255))
grady = abs_sobel_threshold(image, orient='y', sobel_kernel=ksize, thresh=(20, 255))
mag_binary = magnitude_threshold(image, sobel_kernel=ksize, mag_thresh=(20, 255))
dir_binary = direction_threshold(image, sobel_kernel=ksize, thresh=(0, np.pi/2))
plt.imshow(gradx,cmap = 'gray')
plt.show()
plt.imshow(grady,cmap = 'gray')
plt.show()
plt.imshow(mag_binary, cmap = 'gray')
plt.show()
plt.imshow(dir_binary,cmap = 'gray')
plt.show()'''
