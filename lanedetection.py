import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
from moviepy.editor import VideoFileClip
from IPython.display import HTML

def gray(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) #if we use cv2.imread the we will have to use BGR2GRAY to converting to grayscale image
    return gray_image

def gaussian_blur(gray_image, kernel_size):
    blur_image = cv2.GaussianBlur(gray_image, (kernel_size,kernel_size), 0)
    return blur_image

def canny(blur_image,low_threshold,high_threshold):
    edge_image = cv2.Canny(blur_image, low_threshold, high_threshold)
    return edge_image

def region_of_interest(image, edge_image, vertices):
    mask = np.zeros_like(edge_image)
    polygon_colour = 255
    if (len(image.shape) >= 3):
        number_of_colour_channels = image.shape[2]
        ploygon_colour = 255 * number_of_colour_channels
    cv2.fillPoly(mask, vertices, polygon_colour)
    region_of_interest_image = cv2.bitwise_and(edge_image, mask)
    return region_of_interest_image

def region_of_interest_2(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def hough(region_of_interest_image, image, left_lane, right_lane):
    threshold = 15 # basically determines the minimum number of votes required to consider it as a line,
                    #it basically determines how small a line can be detected, if its value is high then smaller lines will be neglected
    minimum_length_of_line = 40 #lines whose lengths are below this value will be rejected
    maximum_line_gap = 20#this is the maximum gap between lines to consider them as a single line
    radial_distance = 2
    theta = (np.pi / 180)
    hough_space_lines = cv2.HoughLinesP(region_of_interest_image, radial_distance, theta, threshold, minLineLength=minimum_length_of_line, maxLineGap=maximum_line_gap) # here I am using probablistic Hough transform
    blank_image_with_lines = np.zeros_like(image)
    for lines in hough_space_lines:                                                                                                                       # as it is computationally more fast
        for x1,y1,x2,y2 in lines:
            slope = (y2 - y1) / (x2 - x1)
            if (slope < 0):
                left_lane.append([x1,y1,x2,y2])
            else:
                right_lane.append([x1,y1,x2,y2])
            cv2.line(blank_image_with_lines,(x1,y1),(x2,y2),(255,0,0),10)
    return (blank_image_with_lines, left_lane, right_lane)

def hough_2(image, region_of_interest_image, rho, theta, threshold, minimum_length_of_line, maximum_line_gap,left_lane,right_lane):
    radial_distance = 2
    hough_space_lines = cv2.HoughLinesP(region_of_interest_image, radial_distance, theta, threshold, minLineLength=minimum_length_of_line, maxLineGap=maximum_line_gap) # here I am using probablistic Hough transform
    blank_image_with_lines = np.zeros_like(image)
    for lines in hough_space_lines:                                                                                                                       # as it is computationally more fast
        for x1,y1,x2,y2 in lines:
            if(x2 != x1):
                slope = (y2 - y1) / (x2 - x1)
                if (slope < 0):
                    left_lane.append([x1,y1,x2,y2])
                else:
                    right_lane.append([x1,y1,x2,y2])
            cv2.line(blank_image_with_lines,(x1,y1),(x2,y2),(255,0,0),10)
    return (blank_image_with_lines, left_lane, right_lane)

def weight(image, Hought_image, weight_1, weight_2, additionalterm):
    weighted_image = cv2.addWeighted(image, weight_1, Hought_image, weight_2, additionalterm)
    return weighted_image

#reading image
image = mpimg.imread('C:/test/test/images/solidWhiteCurve.jpg')
#image = cv2.imread('C:/test/test.jpg')  #alternatively we can use this function as well to read the image
print("image type",type(image),"\n image shape",image.shape)

#displaying the original image
plt.title("original image")
plt.imshow(image)
plt.show()

#converting to gray_scale
gray_image = gray(image)

#displaying the grayscale image
plt.title("grayscale image")
#plt.imshow(gray_image) #doesnot work properly
plt.imshow(gray_image,cmap='gray')
plt.show()   #this line holds the image on the screen

#we need to apply gaussain blurring before applying Canny eventhough Canny function applies its own blur
# read https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html for blurring
kernel_size = 3
blur_image = gaussian_blur(gray_image, kernel_size)

#applying Canny
#according to Canny the ratio of upper_threshold to lower_thereshold should be around 2:1 or 3:1
#read https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/canny_detector/canny_detector.html
low_threshold = 200
high_threshold = 255
edge_image = canny(blur_image,low_threshold,high_threshold)

#displaying the Canny edge blur_image
plt.title("edge image")
plt.imshow(edge_image)
plt.show()

#specifying the region of interest
point_1 = [190,image.shape[0]]
point_2 = [460,300]
point_3 = [540,300]
point_4 = [900,image.shape[0]]
vertices = np.array([[point_1, point_2, point_3, point_4]], dtype=np.int32) #to understand the vertices naming and the general errors in cv2.polyfill
                                                                            #refer http://www.programmersought.com/article/9565947536/
region_of_interest_image = region_of_interest(image, edge_image, vertices)

#displaying the region_of_interest_image
plt.title("region_of_interest_image")
plt.imshow(region_of_interest_image)
plt.show()

#doing Hough Transform
# read the document on the given link to understand what hough transform is doing https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
left_lane = []
right_lane = []
Hough_image, left_lane, right_lane = hough(region_of_interest_image, image,left_lane, right_lane)
blank_image = np.zeros_like(image)
max_y = 0
min_y = 500
max_x = 0
min_x = 800
slope = []
for x1,y1,x2,y2 in left_lane:
    temp = (y2 - y1) / (x2 - x1)
    if (x2 != x1):
        slope.append(temp)
    if (y2 < min_y):
        min_y = y2
    if (y1 > max_y):
        max_y = y1
    if (x2 > max_x):
        max_x = x2
    if (x1 < min_x):
        min_x = x1
print(slope)
mean = np.mean(slope)
print(mean)
print(min_x, max_y)
print(max_x,min_y)
real_y_max = image.shape[0] - min_y
real_y_min = image.shape[0] - max_y
real_x_max = max_x
real_x_min = min_x
b = real_y_min - ((-mean) * real_x_min)
real_y_min = 0
real_x_min = (-b) / (-mean)
print(real_x_min)
cv2.line(blank_image,(int(real_x_min),540),(int(real_x_max),(540 - int(real_y_max))),[255,0,0],10)
slope = []
max_x = 0
min_x = 800
max_y = 0
min_y = 800
for x1,y1,x2,y2 in right_lane:
    temp = (y2 - y1) / (x2 - x1)
    slope.append(temp)
    if (y2 > max_y):
        max_y = y2
    if (y1 < min_y):
        min_y = y1
    if (x2 > max_x):
        max_x = x2
    if (x1 < min_x):
        min_x = x1
mean = np.mean(slope)
mean = -mean
first_y =  image.shape[0] - min_y
second_y =  image.shape[0] - max_y
second_x = max_x
first_x = min_x
b = first_y - ((mean) * first_x)
cv2.line(blank_image,(int(first_x),(540 - int(first_y))),(int(second_x),(540 - int(second_y))),[255,0,0],10)

#displaying Hough Transform image
plt.title("Hough Transform image")
plt.imshow(Hough_image)
plt.show()

#displaying the image with bold lines on the left_lane
plt.title("bold lines on the lane")
plt.imshow(blank_image)
plt.show()

#drawing the Hough Transform lines on the blur_image
weight_1 = 0.8
weight_2 = 1.0
additionalterm = 0.0
print(Hough_image.shape)
weighted_image = weight(image, Hough_image, weight_1, weight_2, additionalterm)
weighted_image_with_bold_lines = weight(image, blank_image, weight_1, weight_2, additionalterm)
#displaying the weighted blur_image
plt.title("weighted image")
plt.imshow(weighted_image)
plt.show()

#displaying the weighted blur_image with bold lines on the lane
plt.title("weighted image_2")
plt.imshow(weighted_image_with_bold_lines)
plt.show()

def process_image(image):
    #converting to gray_scale
    gray_image = gray(image)


    #we need to apply gaussain blurring before applying Canny eventhough Canny function applies its own blur
    # read https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html for blurring
    kernel_size = 5
    blur_image = gaussian_blur(gray_image, kernel_size)

    #applying Canny
    low_threshold = 200
    high_threshold = 250
    edge_image = canny(blur_image, low_threshold, high_threshold)

    #specifying the region of interest
    point_1 = [120,image.shape[0]]
    point_2 = [450,325]
    point_3 = [520,325]
    point_4 = [900,image.shape[0]]
    '''
    point_1 = [190,image.shape[0]]
    point_2 = [630,400]
    point_3 = [750,400]
    point_4 = [1005,image.shape[0]]'''
    vertices = np.array([[point_1, point_2, point_3, point_4]], dtype=np.int32) #to understand the vertices naming and the general errors in cv2.polyfill
                                                                            #refer http://www.programmersought.com/article/9565947536/
    region_of_interest_image = region_of_interest_2(edge_image, vertices)

    #doing Hough Transform
    # read the document on the given link to understand what hough transform is doing https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html
    threshold = 2 # basically determines the minimum number of votes required to consider it as a line,
                    #it basically determines how small a line can be detected, if its value is high then smaller lines will be neglected
    minimum_length_of_line = 10 #lines whose lengths are below this value will be rejected
    maximum_line_gap = 10 #this is the maximum gap between lines to consider them as a single line
    radial_distance = 2
    theta = (np.pi / 180)
    left_lane = []
    right_lane = []
    Hough_image,left_lane,right_lane = hough_2(image,region_of_interest_image, radial_distance, theta, threshold, minimum_length_of_line, maximum_line_gap,left_lane,right_lane)

    #calculations for displaying bold lines on the lanes
    blank_image = np.zeros_like(image)
    max_y = 0
    min_y = 800
    max_x = 0
    min_x = 800
    slope = []
    x_1 = []
    x_2 = []
    for x1,y1,x2,y2 in left_lane:
        if(x2 != x1):
            temp = (y2 - y1) / (x2 - x1)
            slope.append(temp)
            if (y2 < min_y):
                min_y = y2
            if (y1 > max_y):
                max_y = y1
            if (x2 > max_x and x2 < 470):
                max_x = x2
                #x_1.append(max_x)
            if (x1 < min_x and x1 < 400):
                min_x = x1
                #x_2.append(min_x)
    #print(slope)
    mean = np.mean(slope)
    mean = mean
    real_x1 = min_x
    real_x2 = max_x
    real_y1 = 540 - max_y
    real_y2 = 540 - min_y
    b = real_y1 - ((-mean) * real_x1)
    real_y1 = 0
    real_x1 = (-b) / (-mean)
    cv2.line(blank_image,(int(real_x1),(540 - 0)),(int(real_x2),(540 - int(real_y2))),[255,0,0],10)
    slope = []
    max_x = 0
    min_x = 800
    max_y = 0
    min_y = 800
    for x1,y1,x2,y2 in right_lane:
        temp = (y2 - y1) / (x2 - x1)
        if (x2 != x1):
            slope.append(temp)
            if (y2 > max_y):
                max_y = y2
            if (y1 < min_y):
                min_y = y1
            if (x2 > max_x and x2 > 610 and x2 < 870):
                max_x = x2
            if (x1 < min_x and x1 > 470):
                min_x = x1
    mean = np.mean(slope)
    if(mean==0):
        mean=1
    real_x1 = min_x
    real_x2 = max_x
    real_y1 = 540 - min_y
    real_y2 = 540 - max_y
    b = real_y2 - ((-mean) * real_x2)
    real_y2 = 0
    real_x2 = (-b) / (-mean)
    cv2.line(blank_image,(int(real_x1),(540 - int(real_y1))),(int(real_x2),(540 - int(real_y2))),[255,0,0],10)
    #NOTE
    #in the left_lane real_x1 and real_y1 are the co-ordinates of the bottom point in our normal x-y co-ordinate system
    #in the right_lane real_x1 and real_y1 are the co-ordinates of the top point in our normal x-y co-ordinate system

    #drawing the Hough Transform lines on the blur_image
    weight_1 = 0.8
    weight_2 = 1
    additionalterm = 0.0
    #print(Hough_image.shape)
    weighted_image = weight(image, Hough_image, weight_1, weight_2, additionalterm)
    #weighted_image_with_bold_lines = weighted_img(image, blank_image, weight_1, weight_2, additionalterm)
    return weighted_image #weighted_image_with_bold_lines

white_output = 'C:/test/test/video/output_SolidYellowLeft.mp4'
#clip1 = VideoFileClip('C:/test/project_video.mp4').subclip(0,1)
clip1 = VideoFileClip('C:/test/test/video/SolidYellowLeft.mp4')
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)
