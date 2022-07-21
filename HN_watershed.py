# Source (watershed tutorial on coins.jpg): https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


# for image correction: 
from skimage import io
# import numpy as np


#variables: kernel size, iterations (closing, dilation), distance transform factor
# closing > opening, tree canopy classified as marker 1.00

# Notes:
# Parameter adjustment will likely be flight specific (seems to potentially be correlated with time of day/shadows)
# Edge clumps are often the worst segmented; orthomosaic may help in this but unclear, may segment images further
# Tree crowns are marked as 1.00 (background), but workable
# Will finalize the segmented images, assign Ohia trees, then use as ML input
# Starting on Friday, will set-up the ML architecture

# correct for illumination
# Kalman filter



flight = 4 # Change for flight number (progresses with time of day)
flight_files = ['0591', '0628', '0696', '0706']
img_file = rf'C:\Users\haley\UROPSP22\vision-main\torchvision\models\test_img\DJI_{flight_files[flight-1]}.JPG' # file image (non-orthomosaic) from the given flight

original_img = cv.imread(img_file)

#adjust range for given parameter

def trials(img):
    for i in range(1, 7): # original: 1, 7
        # img = cv.imread(img_file)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

        # noise removal
        kernel = np.ones((3,3),np.uint8) # original: 3,3, trying 1-5
        closing = cv.morphologyEx(thresh,cv.MORPH_CLOSE,kernel, iterations = 2) # original: 2

        # sure background area
        sure_bg = cv.dilate(closing,kernel,iterations=i) # original: 3

        # Finding sure foreground area
        dist_transform = cv.distanceTransform(closing,cv.DIST_L2,5)
        ret, sure_fg = cv.threshold(dist_transform,0.1*7*dist_transform.max(),255,0) # original: 0.7
        
        # Finding unknown region
        sure_fg = np.uint8(sure_fg)
        unknown = cv.subtract(sure_bg,sure_fg)

        # Marker labelling
        ret, markers = cv.connectedComponents(sure_fg)
        # Add one to all labels so that sure background is not 0, but 1
        markers = markers+1
        # Now, mark the region of unknown with zero
        markers[unknown==255] = 0

        markers = cv.watershed(img,markers)
        img[markers == -1] = [255,0,0] #to show original image

        plt.subplot(3, 2, i),plt.imshow(markers)
        plt.title(f'Dilation Iterations: {i}'), plt.xticks([]), plt.yticks([])
        # plt.subplot(3, 2, 2+2*(i-1)),plt.imshow(markers)
        # plt.title(f'Distance transform: {i}'), plt.xticks([]), plt.yticks([])
    plt.show()


def segmentation(img, dist_transform_factor):
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3,3),np.uint8) # original: 3,3, trying 1-5
    closing = cv.morphologyEx(thresh,cv.MORPH_CLOSE,kernel, iterations = 2) # original: 2

    # sure background area
    sure_bg = cv.dilate(closing,kernel,iterations=3) # original: 3

    # Finding sure foreground area
    dist_transform = cv.distanceTransform(closing,cv.DIST_L2,5)
    ret, sure_fg = cv.threshold(dist_transform,0.1*dist_transform_factor*dist_transform.max(),255,0) # original: 0.7
    
    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv.connectedComponents(sure_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0

    markers = cv.watershed(img,markers)
    img[markers == -1] = [255,0,0] #to show original image

    return markers, img

    # plt.subplot(1, 2, 1),plt.imshow(img)
    # plt.title(f'Flight: {flight}'), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 2, 2),plt.imshow(markers)
    # plt.title(f'Flight: {flight}'), plt.xticks([]), plt.yticks([])
    # plt.show()

####################
# Image Correction #
####################

def gamma_correction(img):
    # nat = io.imread(img_file)
    nat = img
    nat_2 = cv.cvtColor(nat, cv.COLOR_BGR2RGB)

    # cv2_imshow(nat_2)

    def gammaCorrection(src, gamma):
        invGamma = 1 / gamma

        table = [((i / 255) ** invGamma) * 255 for i in range(256)]
        table = np.array(table, np.uint8)

        return cv.LUT(src, table)


    gamma = 2.5      # change the value here to get different result
    adjusted = gammaCorrection(nat_2, gamma=gamma)
    # cv2_imshow(adjusted)

    # plt.subplot(1, 2, 1),plt.imshow(nat_2) 
    # plt.title('original image'), plt.xticks([]), plt.yticks([])
    # plt.subplot(1, 2, 2),plt.imshow(adjusted) 
    # plt.title('corrected image'), plt.xticks([]), plt.yticks([])

    # plt.show()
    return adjusted



def main():
    # img = gamma_correction(original_img)
    # trials(img)

    og_marker, og_seg = segmentation(original_img)
    corrected = gamma_correction(original_img)
    cor_marker, cor_seg = segmentation(corrected)

    plt.subplot(2, 2, 1),plt.imshow(og_marker) 
    plt.title('Original Image Markers'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 2),plt.imshow(og_seg) 
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 3),plt.imshow(cor_marker) 
    plt.title('Corrected Image Markers'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 2, 4),plt.imshow(cor_seg) 
    plt.title('Corrected Image'), plt.xticks([]), plt.yticks([])

    plt.show()

# main()