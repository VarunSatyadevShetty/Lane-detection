import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import math
import pickle
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def color_thresholding(image,ksize,thresh):
    new_image_hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    h_channel = new_image_hls[:,:,0]
    l_channel = new_image_hls[:,:,1]
    s_channel = new_image_hls[:,:,2]
    binary_ls_channel_thresholding = np.zeros_like(s_channel)
    temp_4 = (s_channel >120) & (l_channel >40)
    temp_5 = l_channel >205
    binary_ls_channel_thresholding[temp_4|temp_5] = 1
    return binary_ls_channel_thresholding

def gradient_thresholding(image,ksize,thresh):
    gray_scale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobel_x = cv2.Sobel(gray_scale,cv2.CV_64F,1,0,ksize)
    sobel_y = cv2.Sobel(gray_scale,cv2.CV_64F,0,1,ksize)
    combined = np.sqrt((sobel_x*sobel_x)+(sobel_y*sobel_y))
    abs_sobel_x = np.absolute(sobel_x)
    abs_sobel_y = np.absolute(sobel_y)
    gradient = np.arctan2(abs_sobel_y,abs_sobel_x)
    abs_sobel_combined = np.absolute(combined)
    #the below step brings the values in the range of (0,255)
    new_abs_sobel_x = np.uint8(255 * (abs_sobel_x/np.max(abs_sobel_x)))
    new_abs_sobel_y = np.uint8(255 * (abs_sobel_y/np.max(abs_sobel_y)))
    new_abs_sobel_combined = np.uint8(255 * (abs_sobel_combined/np.max(abs_sobel_combined)))
    temp_1 = None
    temp_2 = None
    temp_3 = None
    temp_4 = None
    temp_5 = None
    temp_1 = new_abs_sobel_x > 40
    temp_2 = new_abs_sobel_combined > 70
    temp_3 = (gradient > 0.7) & (gradient < 1.4)
    binary_sobel_thresholding = np.zeros_like(new_abs_sobel_x)
    binary_sobel_thresholding[temp_1&temp_2&temp_3] = 1
    return binary_sobel_thresholding
