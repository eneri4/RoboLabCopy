# import libraries
import cv2 as cv
import numpy as np
import os

file_path = r"C:\Users\ihong\OneDrive - Olin College of Engineering\Olin 2024-2025\RoboLab\Field Nav\phoenixbot_data\0B9odeEn.jpg"

assert os.path.exists(file_path), "File not found"

# blur images
img = cv.imread(file_path)
cv.imshow("Original Image", img)
hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
hue_img = hsv_img[:, :, 1]
cv.imshow("HSV Image", hsv_img)
cv.imshow("Saturation channel", hue_img)
assert img is not None, "file could not be read, check with os.path.exists()" #ensure image exists
blur = cv.GaussianBlur(hue_img,(5,5),0)
 
# find normalized_histogram, and its cumulative distribution function
hist = cv.calcHist([blur],[0],None,[256],[0,256])
hist_norm = hist.ravel()/hist.sum()
Q = hist_norm.cumsum()
 
bins = np.arange(256)
 
fn_min = np.inf
thresh = -1
 
for i in range(1,256):
 p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
 q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
 if q1 < 1.e-6 or q2 < 1.e-6:
    continue
 b1,b2 = np.hsplit(bins,[i]) # weights
 
 
 # finding means and variances
 m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
 v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2
 
 # calculates the minimization function
 fn = v1*q1 + v2*q2
 if fn < fn_min:
    fn_min = fn
    thresh = i
 
# find otsu's threshold value with OpenCV function
ret, otsu = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
cv.imshow("ret",otsu)
print( "{} {}".format(thresh,ret) )

masked = cv.bitwise_and(img, img, mask=otsu)
cv.imshow("Thresholded Image", masked)
cv.waitKey(0)
cv.destroyAllWindows()

# kenneth suggestion: morphological transformations to make image cleaner
# kenneth tip 1: always visualize results
# kenneth tip 2: make open source library of real data to train alg

# change code to use video as input 

# kenneth tip 3: using opencv to do like 80% of annotations for you??