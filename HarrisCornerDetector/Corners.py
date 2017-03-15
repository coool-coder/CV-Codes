#!/usr/bin/env python
import cv2
import numpy as np
import sys
import getopt
import operator
from matplotlib import pyplot as plt
import math

np.seterr(divide='ignore', invalid='ignore')
def readImage(filename):  
    img = cv2.imread(filename, 0)
    if img is None:
        print('Invalid image:' + filename)
        return None
    else:
        print('Image successfully read...')
        return img

# def quantization():

def NonMaxSuppression(R):
	for i in range(1, R.shape[0]-1):
		for j in range(1, R.shape[1]-1):
			w = R[i-1:i+2, j-1:j+2]
			a = angle[i-1:i+2, j-1:j+2]			
			flag = 0
			if angle[i,j] == 0:
				if R[i,j] >= R[i,j+1] and R[i,j] >= R[i,j-1]:
					R[i,j]=1
				else:
					R[i,j]=0

			elif angle[i,j] == 45:
				if R[i,j] >= R[i-1,j+1] and R[i,j] >= R[i+1,j-1]:
					R[i,j]=1
				else:
					R[i,j]=0
				
			elif angle[i,j] == 90:
				if R[i,j] >= R[i-1,j] and R[i,j] >= R[i+1,j]:
					R[i,j]=1
				else:
					R[i,j]=0

			elif angle[i,j] == 135:
				if R[i,j] >= R[i-1,j-1] and R[i,j] >= R[i+1,j+1]:
					R[i,j]=1
				else:
					R[i,j]=0

	cv2.imshow("suppresssed", R)

def findCorners(img, window_size, k, thresh):
    dy, dx = np.gradient(img)				# finding gradient of image ... dy - gradient in y direction and dx -> gradient in x direction.
    Ixx = dx**2
    Ixy = dy*dx
    Iyy = dy**2
    height = img.shape[0]
    width = img.shape[1]
    global angle
    angle = (np.arctan(dy/dx))*180/math.pi
    for i in range(angle.shape[0]):
    	for j in range(angle.shape[1]):
    		if -22.5<=angle[i,j] and angle[i,j]<=22.5:
    			angle[i,j] = 0
    		elif 22.5<=angle[i,j] and angle[i,j]<=67.5:
    			angle[i,j] = 45
    		elif 67.5<=angle[i,j] and angle[i,j]<=90:
    			angle[i,j] = 90
    		elif -67.5<=angle[i,j]and angle[i,j]<= -22.5:
    			angle[i,j] = 135
    		elif -90<=angle[i,j] and angle[i,j]<=-67.5:
    			angle[i,j] = 90
    		# elif 202.5<=angle[i,j] and angle[i,j]<=247.5:
    		# 	angle[i,j] = 45
    		# elif 247.5<=angle[i,j]and angle[i,j]<=292.5:
    		# 	angle[i,j] = 90
    		# elif 247.5<=angle[i,j]and angle[i,j]<=292.5:
    		# 	angle[i,j] = 90
    		# elif 292.5<=angle[i,j] and angle[i,j]<=337.5:
    		# 	angle[i,j] = 135
    		
    
    # print(angle)
    R = img.copy()
    cornerList = []
    newImg = img.copy()
    color_img = cv2.cvtColor(newImg, cv2.COLOR_GRAY2RGB)
    global offset
    offset = window_size/2 # 5/2 -> 3

    #Loop through image and find our corners
    print ("Finding Corners...")
    for y in range(offset, height-offset):  # range: 3 -> 
        for x in range(offset, width-offset):
            #Calculate sum of squares
            windowIxx = Ixx[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIxy = Ixy[y-offset:y+offset+1, x-offset:x+offset+1]
            windowIyy = Iyy[y-offset:y+offset+1, x-offset:x+offset+1]
            Sxx = windowIxx.sum()
            Sxy = windowIxy.sum()
            Syy = windowIyy.sum()

            #Find determinant and trace, use to get corner response
            det = (Sxx * Syy) - (Sxy**2)
            trace = Sxx + Syy
            r = det - k*(trace**2)
            R[y,x] = r
            #If corner response is over threshold, color the point and add to corner list
            if r > thresh:
            	
            	cornerList.append([x, y, r])
            	color_img.itemset((y, x, 0), 0)
            	color_img.itemset((y, x, 1), 0)
            	color_img.itemset((y, x, 2), 255)
    # R = R.astype(np.uint8)
    # plt.hist(R.ravel(), R.max(), [0,R.max()]); plt.show() 
    cv2.imshow("R", R)
    NonMaxSuppression(R)
    


    return color_img, cornerList



def main():   
    
    print (" Enter the Image Name :  ")
    img_name=str(raw_input())

    print (" Enter the Window Size : Default: 5 ")
    window_size=int(raw_input())

    print ("Enter K Corner Response : value of K Default: 0.04 ")
    k=float(raw_input())

    print ("Enter Threshold Default: 10000")
    thresh=raw_input()

    img = readImage(img_name)

    if img is not None:
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if len(img.shape) == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        finalImg, cornerList = findCorners(img, int(window_size), float(k), int(thresh))
        if finalImg is not None:
            cv2.imwrite("finalimage1.png", finalImg)

        # Write top 100 corners to file
        cornerList.sort(key=operator.itemgetter(2))
        outfile = open('corners.txt', 'w')
        for i in range(100):
            outfile.write(str(cornerList[i][0]) + ' ' + str(cornerList[i][1]) + ' ' + str(cornerList[i][2]) + '\n')
        outfile.close()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
