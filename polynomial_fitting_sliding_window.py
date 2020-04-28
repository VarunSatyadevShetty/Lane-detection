import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import math
import pickle
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def region_of_interest(image,combined_image,vertices):
    mask = np.zeros_like(combined_image)
    polygon_colour = 255
    if (len(image.shape) >= 3):
        number_of_colour_channels = image.shape[2]
        ploygon_colour = 255 * number_of_colour_channels
    cv2.fillPoly(mask, vertices, polygon_colour)
    region_of_interest_image = cv2.bitwise_and(combined_image, mask)
    return region_of_interest_image

def region_of_interest_2(image,vertices):
    mask = np.zeros_like(image)
    polygon_colour = 255
    if (len(image.shape) >= 3):
        number_of_colour_channels = image.shape[2]
        ploygon_colour = 255 * number_of_colour_channels
    cv2.fillPoly(mask, vertices, polygon_colour)
    region_of_interest_image = cv2.bitwise_and(image, mask)
    return region_of_interest_image

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

def calculating_undistorted_image(image, objpoints, imgpoints):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image.shape[1::-1],None,None)
    dst = cv2.undistort(image, mtx, dist, None,mtx)
    return dst

def finding_lane(image):
    #finding the histogram of th region of most probability of finding the lane
    histogram = np.sum(image[image.shape[0]//2:,:],axis=0)
    out_image = np.dstack((image,image,image))
    midpoint = np.int(histogram.shape[0]//2)
    left_lane_base = np.argmax(histogram[:midpoint])
    right_lane_base = np.argmax(histogram[midpoint:]) + midpoint
    number_of_windows = 9
    width = 100
    margin = 100
    height = np.int(image.shape[0]//number_of_windows)
    min_no_of_pixels_required_to_recenter_the_image = 50
    non_zero = image.nonzero()
    nonzeroy = np.array(non_zero[0])
    nonzerox = np.array(non_zero[1])
    left_current = left_lane_base
    right_current = right_lane_base
    left = []
    right = []
    for i in range(number_of_windows):
        left_x1 = (left_current - width)
        left_x2 = (left_current + width)
        left_y1 = (image.shape[0] - ((i+1)*height))
        left_y2 = (image.shape[0] - ((i)*height))
        right_x1 = (right_current - width)
        right_x2 = (right_current + width)
        right_y1 = (image.shape[0] - ((i+1)*height))
        right_y2 = (image.shape[0] - ((i)*height))

        # Draw the windows on the visualization image
        cv2.rectangle(out_image,(left_x1,left_y1),
        (left_x2,left_y2),(0,255,0), 2)
        cv2.rectangle(out_image,(right_x1,right_y1),
        (right_x2,right_y2),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window #
        indices_left = ((nonzeroy>=left_y1)&(nonzeroy<left_y2)&(nonzerox>=left_x1)&(nonzerox<left_x2)).nonzero()[0]
        indices_right = ((nonzeroy>=right_y1)&(nonzeroy<right_y2)&(nonzerox>=right_x1)&(nonzerox<right_x2)).nonzero()[0]
        left.append(indices_left)
        right.append(indices_right)
        if (len(indices_left) > min_no_of_pixels_required_to_recenter_the_image):
            left_current = np.int(np.mean(nonzerox[indices_left]))
        if (len(indices_right) > min_no_of_pixels_required_to_recenter_the_image):
            right_current = np.int(np.mean(nonzerox[indices_right]))
    try:
        left = np.concatenate(left)
        right = np.concatenate(right)
    except ValueError:

        pass
    left_x = nonzerox[left]
    left_y = nonzeroy[left]
    right_x = nonzerox[right]
    right_y = nonzeroy[right]
    return left_x,left_y,right_x,right_y,out_image

def polynomial_fitting(image):
    left_x,left_y,right_x,right_y,out_image = finding_lane(image)
    '''plt.imshow(out_image)
    plt.show()'''
    #print(left_x,left_y,right_x,right_y)
    constant_1 = np.polyfit(left_y,left_x,2)
    constant_2 = np.polyfit(right_y,right_x,2)
    ploty = np.linspace(0,image.shape[0]-1,image.shape[0]) #(start,end,number)
    try:
        left_fit_x = (constant_1[0]*ploty**2) + (constant_1[1]*ploty) + (constant_1[2])
        right_fit_x = (constant_2[0]*ploty**2) + (constant_2[1]*ploty) + (constant_2[2])
    except TypeError:
        print('failed to fit line')
        left_fit_x = (ploty**2) + (ploty)
        right_fit_x = (ploty**2) + (ploty)

    out_image[left_y,left_x] = [255,0,0]
    out_image[right_y,right_x] = [0,0,255]
    '''plt.imshow(out_image)
    plt.show()'''
    plt.plot(left_fit_x,ploty,color='yellow')
    plt.plot(right_fit_x,ploty,color='yellow')
    plt.title("sliding window visualization")
    plt.imshow(out_image)
    plt.show()
    #plt.show()
    return out_image,constant_1,constant_2,ploty


dist_pickle = pickle.load( open( 'C:/test/wide_dist_pickle.p', "rb" ) )
objpoints = dist_pickle["objpoints"]
imgpoints = dist_pickle["imgpoints"]
image = cv2.imread('C:/test/perspective_transform/test5.jpg')
plt.title("original image")
plt.imshow(image)
plt.show()
dst = calculating_undistorted_image(image, objpoints, imgpoints)
thresh = (15,255)
ksize = 5
new_image_1,new_image_2 = combination(dst,ksize,thresh)
'''plt.imshow(new_image_2,cmap = 'gray')
plt.show()'''
point_1 = [140,700]
point_2 = [510,445]
point_3 = [740,445]
point_4 = [1120,700]
vertices = np.array([[point_1, point_2, point_3, point_4]], dtype=np.int32)
region_of_interest_image = region_of_interest(dst,new_image_2,vertices)
#plt.imshow(region_of_interest_image,cmap = 'gray')
#plt.show()
#plt.imshow(region_of_interest_image,cmap='gray')
src = np.float32([[140,700],[560,445],[740,445],[1120,700]])
desired =np.float32([[400,700],[400,100],[950,100],[950,700]])
#plt.plot(140,700,'.')
#plt.plot(560,445,'.')
#plt.plot(740,445,'.')
#plt.plot(1120,700,'.')
#plt.show()
M = cv2.getPerspectiveTransform(src,desired)
wraped = cv2.warpPerspective(new_image_2,M,(new_image_2.shape[1],new_image_2.shape[0]),flags=cv2.INTER_LINEAR)
'''plt.imshow(wraped,cmap='gray')
plt.show()'''
point_1 = [350,710]
point_2 = [350,100]
point_3 = [970,100]
point_4 = [970,710]
vertices = np.array([[point_1, point_2, point_3, point_4]], dtype=np.int32)
region_of_interest_image_2 = region_of_interest_2(wraped,vertices)
'''plt.imshow(region_of_interest_image_2)
plt.show()'''
gray = np.zeros_like(wraped)
gray[(region_of_interest_image_2 > 0)] = 255
'''plt.imshow(gray)
plt.show()'''
#print(gray.shape)
#img = fit_polynomial(gray)
image_1,left_fit_x,right_fit_x,plot_y = polynomial_fitting(gray)
'''plt.imshow(image_1)
plt.show()'''
###
margin = 100
left_fit = (left_fit_x)
right_fit = (right_fit_x)
ploty = [plot_y]

def polynomial_fitting_2(image,left_x,left_y,right_x,right_y):
    constant_1 = np.polyfit(left_y,left_x,2)
    constant_2 = np.polyfit(right_y,right_x,2)
    ploty = np.linspace(0,image.shape[0]-1,image.shape[0]) #(start,end,number)
    try:
        left_fit_x = (constant_1[0]*ploty**2) + (constant_1[1]*ploty) + (constant_1[2])
        right_fit_x = (constant_2[0]*ploty**2) + (constant_2[1]*ploty) + (constant_2[2])
    except TypeError:
        print('failed to fit line')
        left_fit_x = (ploty**2) + (ploty)
        right_fit_x = (ploty**2) + (ploty)
    #plt.show()
    return left_fit_x,right_fit_x,ploty,constant_1,constant_2

def search_around_the_fitted_polynomial(image):
    width = 100
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    '''left = (left_fit[0]*nonzeroy**2) + (left_fit[1]*nonzeroy) + (left_fit[2])
    right = (right_fit[0]*nonzeroy**2) + (right_fit[1]*nonzeroy) + (right_fit[2])
    indices_left = ((nonzerox>(left-width))&(nonzerox<(left+width)))
    indices_right = ((nonzerox>(right-width))&(nonzerox<(right+width)))'''
    indices_left = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    indices_right = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    leftx = nonzerox[indices_left]
    lefty = nonzeroy[indices_left]
    rightx = nonzerox[indices_right]
    righty = nonzeroy[indices_right]
    left_fit_x,right_fit_x,ploty,constant_1,constant_2 = polynomial_fitting_2(image,leftx,lefty,rightx,righty)
    #Visualization
    out_img = np.dstack((image, image, image))*255
    window_img = np.zeros_like(out_img)
    out_img[nonzeroy[indices_left], nonzerox[indices_left]] = [255, 0, 0]
    out_img[nonzeroy[indices_right], nonzerox[indices_right]] = [0, 0, 255]

    left_line_window1 = np.array([np.transpose(np.vstack([left_fit_x-margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fit_x+margin,
                                 ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fit_x-margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fit_x+margin,
                                 ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))
        # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (255,0, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,0, 255))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
    plt.title("sliding window approach")
    # Plot the polynomial lines onto the image
    plt.plot(left_fit_x, ploty, color='yellow')
    plt.plot(right_fit_x, ploty, color='yellow')
    ## End visualization steps ##

    return result
image = search_around_the_fitted_polynomial(gray)
plt.title("polynomial fitted for lane lines")
plt.imshow(image)
plt.show()
print("hello")
def measure_curvature(gray):
    width = 100
    nonzero = gray.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    indices_left = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
                    left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
                    left_fit[1]*nonzeroy + left_fit[2] + margin)))
    indices_right = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
                    right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
                    right_fit[1]*nonzeroy + right_fit[2] + margin)))
    leftx = nonzerox[indices_left]
    lefty = nonzeroy[indices_left]
    rightx = nonzerox[indices_right]
    righty = nonzeroy[indices_right]
    lefty = lefty*(30/600) # meters per pixel in y dimension
    righty = righty*(30/600)
    leftx = leftx*(3.7/550) # meters per pixel in x dimension
    rightx = rightx*(3.7/550)
    #print(leftx,lefty,rightx,righty)
    left_fit_x,right_fit_x,ploty,constant_1,constant_2 = polynomial_fitting_2(gray,leftx,lefty,rightx,righty)
    y_eval = np.max(plot_y)
    left_curve = ((1 + (2*constant_1[0]*y_eval + constant_1[1])**2)**1.5) / np.absolute(2*constant_1[0])
    right_curve = ((1 + (2*constant_2[0]*y_eval + constant_2[1])**2)**1.5) / np.absolute(2*constant_2[0])
    return left_curve,right_curve
left_measured,right_measured = measure_curvature(gray)
print(left_measured,right_measured)

'''def one(left_y,left_x,right_y,right_x):
    left_after_fitting = np.array([lefty,left_x])
    right_after_fitting = np.array([right_y,right_x])
    return left_after_fitting,right_after_fitting
left_fit_x,right_fit_x,plot_y = polynomial_fitting_2(gray,left_after_fitting[0],left_after_fitting[1],right_after_fitting[0],right_after_fitting[1])
image = search_around_the_fitted_polynomial(gray)
bottom_half = gray[gray.shape[0]//2:,:]
histogram = np.sum(bottom_half, axis=0)
plt.plot(histogram)
plt.show()'''
