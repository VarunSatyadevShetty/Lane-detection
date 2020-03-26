import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import numpy as np
import math
import pickle
from moviepy.editor import VideoFileClip
from IPython.display import HTML


class line():
    left_detected = False
    right_detected = False

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
    combined_binary_image = np.zeros_like(binary_sobel_thresholding)
    combined_binary_image[(binary_sobel_thresholding == 1)|(binary_s_channel_thresholding == 1)] = 1
    return color_image, combined_binary_image

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

    plt.plot(left_fit_x,ploty,color='yellow')
    plt.plot(right_fit_x,ploty,color='yellow')
    #plt.show()
    return out_image,constant_1,constant_2,ploty,left_fit_x,right_fit_x

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

def search_around_the_fitted_polynomial(image,left_fit,right_fit):
    margin = 100
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
    '''out_img = np.dstack((image, image, image))*255
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
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)'''

    return left_fit_x,right_fit_x,ploty,constant_1,constant_2

def measure_curvature(gray,left_fit,right_fit):
    margin = 100
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
    y_eval = np.max(ploty)
    left_curve = ((1 + (2*constant_1[0]*y_eval + constant_1[1])**2)**1.5) / np.absolute(2*constant_1[0])
    right_curve = ((1 + (2*constant_2[0]*y_eval + constant_2[1])**2)**1.5) / np.absolute(2*constant_2[0])
    return left_curve,right_curve

def process_image(image_original):
    #undistorting the frame of image
    undistorted_image = calculating_undistorted_image(image_original, object_points, image_points)
    #obtaining the binary image
    thresh = (15,255)
    ksize = 5
    color_image,combined_binary_image = combination(undistorted_image,ksize,thresh)
    #doing perspective transform
    src = np.float32([[140,700],[560,445],[740,445],[1120,700]])
    desired =np.float32([[400,700],[400,100],[950,100],[950,700]])
    M_1 = cv2.getPerspectiveTransform(src,desired)
    wraped = cv2.warpPerspective(combined_binary_image,M_1,(combined_binary_image.shape[1],combined_binary_image.shape[0]),flags=cv2.INTER_LINEAR)
    #filtering the wraped(perspective transformed) image
        #step one by using a region of interest mask
    point_1 = [350,710]
    point_2 = [350,100]
    point_3 = [970,100]
    point_4 = [970,710]
    vertices = np.array([[point_1, point_2, point_3, point_4]], dtype=np.int32)
    region_of_interest_image_2 = region_of_interest_2(wraped,vertices)
        #doing thresholding
    gray = np.zeros_like(wraped)
    gray[(region_of_interest_image_2 > 0)] = 255
    #doing smart search
    if (line.left_detected == True and line.right_detected == True):
        left_fit_x,right_fit_x,ploty,constant_1,constant_2 = search_around_the_fitted_polynomial(gray,constant_1,constant_2)
    else:
        out_image,constant_1,constant_2,ploty,left_fit_x,right_fit_x = polynomial_fitting(gray)
        left_x,left_y,right_x,right_y,out_image = finding_lane(gray)
    if (len(left_x)>0 and len(left_y)>0):
        line.left_detected = True
    elif (len(right_x)>0 and len(right_y)>0):
        line.right_detected = True
    else:
        line.left_detected = False
        line.right_detected = False
    #left_x,left_y,right_x,right_y,out_image = finding_lane(gray)
    '''#doing polynomial fitting using sliding window method
    out_image_from_polynomial_fitting,constant_1,constant_2,ploty = polynomial_fitting(gray)
    #doing smart search using the line co-ordinates from the previous frame
    image = search_around_the_fitted_polynomial(gray,constant_1,constant_2)'''
    #measuring the radius of curvature
    left_measured,right_measured = measure_curvature(out_image,constant_1,constant_2)
    print(left_measured,right_measured)
    #unwraping the image
    M_2 = cv2.getPerspectiveTransform(desired,src)
    unwraped = cv2.warpPerspective(combined_binary_image,M_2,(combined_binary_image.shape[1],combined_binary_image.shape[0]),flags=cv2.INTER_LINEAR)
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(unwraped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fit_x, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fit_x,ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, M_2, (image_original.shape[1], image_original.shape[0]))
    # Combine the result with the original image
    result = cv2.addWeighted(undistorted_image, 1, newwarp, 0.3, 0)
    return result
###extracting the object_points and the image_points which is used to caliberate the camera
dist_pickle = pickle.load( open( 'C:/test/wide_dist_pickle.p', "rb" ) )
object_points = dist_pickle["objpoints"]
image_points = dist_pickle["imgpoints"]
line = line()
### video
white_output = 'C:/test/project_video.mp4'
clip1 = VideoFileClip('C:/test/project_video.mp4').subclip(0,5)
#clip1 = VideoFileClip('C:/test/project_video.mp4')
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)
