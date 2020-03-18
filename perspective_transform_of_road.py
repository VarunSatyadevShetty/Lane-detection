import numpy as np
import cv2
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

def region_of_interest(image,combined_image,vertices):
    mask = np.zeros_like(combined_image)
    polygon_colour = 255
    if (len(image.shape) >= 3):
        number_of_colour_channels = image.shape[2]
        ploygon_colour = 255 * number_of_colour_channels
    cv2.fillPoly(mask, vertices, polygon_colour)
    region_of_interest_image = cv2.bitwise_and(combined_image, mask)
    return region_of_interest_image

def undistorted(image,objpoints,imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image.shape[1::-1],None,None)
    dst = cv2.undistort(image, mtx, dist, None,mtx)
    return dst

def combination_thresholding(image,ksize,thresh):
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

images = glob.glob('C:/test/perspective_transform/test*.jpg')
objectpoints = []
imagepoints = []
objectp = np.zeros((4,3), np.float32)
objectp[:,:2] = np.mgrid[0:2,0:2].T.reshape(-1,2)
print(objectp)
for image in images:
    ksize = 5
    thresh = (15,255)
    image = cv2.imread(image)
    #plt.imshow(image)
    #plt.show()
    color_image,combined_image = combination_thresholding(image,ksize,thresh)
    point_1 = [200,image.shape[0]]
    point_2 = [460,360]
    point_3 = [800,360]
    point_4 = [1100,image.shape[0]]
    vertices = np.array([[point_1, point_2, point_3, point_4]], dtype=np.int32)
    #region_of_interest_image = region_of_interest(image,combined_image,vertices)
    #plt.imshow(region_of_interest_image,cmap = 'gray')
    #plt.show()
    src = np.float32([point_1,point_2,point_3,point_4])
    objectpoints.append(objectp)
    imagepoints.append(src)

image = cv2.imread('C:/test/perspective_transform/test3.jpg')
#vertices = np.array([[point_1, point_2, point_3, point_4]], dtype=np.int32)
#region_of_interest_image = region_of_interest(image,combined_image,vertices)
#objectpoints.append(objectp)
#imagepoints.append(src)
dst = undistorted(image,objectpoints,imagepoints)
plt.imshow(dst)
plt.show()
plt.imshow(image)
src = np.float32([[300,670],[538,466],[680,466],[1115,670]])
desired =np.float32([[400,700],[600,100],[900,100],[1220,700]])
plt.plot(300,670,'.')
plt.plot(538,466,'.')
plt.plot(680,446,'.')
plt.plot(1115,670,'.')
plt.show()
M = cv2.getPerspectiveTransform(src,desired)
wraped = cv2.warpPerspective(dst,M,(dst.shape[1],dst.shape[0]),flags=cv2.INTER_LINEAR)
plt.imshow(wraped)
plt.show()
