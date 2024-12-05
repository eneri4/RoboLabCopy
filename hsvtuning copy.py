# import libraries
import cv2 as cv
import numpy as np
import os

# image input
file_path = r"/home/irene/RoboLabCopy/Field Nav/image (1).webp"

assert os.path.exists(file_path), "File not found"

img = cv.imread(file_path)
cv.imshow("Original Image", img)

# change to HSV
hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
hue_img = hsv_img[:, :, 0]
cv.imshow("HSV Image", hsv_img)
# hue channel
cv.imshow("Hue channel", hue_img)
assert img is not None, "file could not be read, check with os.path.exists()" #ensure image exists
# blur image
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

# masked image
masked = cv.bitwise_and(img, img, mask=otsu)
cv.imshow("Thresholded Image", masked)

# opened image
kernel = cv.getStructuringElement(cv.MORPH_CROSS,(5,5)) # size of kernel determined by amount of noise
#opening = cv.morphologyEx(masked, cv.MORPH_OPEN, kernel)
#cv.imshow("Eroded Image", opening)

cv.waitKey(0)
cv.destroyAllWindows()

# try using simpler morphological operations & combos
# try using gaussian blur, median blur, and bilateral filtering and see their differences


