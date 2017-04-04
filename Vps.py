# (y - y1)(x2-x1) = (y2-y1)(x-x1)
import cv2
import numpy as np
from sympy import solve, Poly, Eq, Function, exp
from sympy.abc import x,u,v
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not

refPt = []
cropping = False

def find_camParams(vp1, vp2, vp3):
	K_inv = np.array([[x,0,-u*(x)],[0, x, -v*(x)],[0,0,1]])  # (x=1/f)
	pr = np.dot(K_inv.T, K_inv)
	#equations
	eq1 = np.dot(vp1, np.dot(pr, vp1.T))
	eq2 = np.dot(vp2, np.dot(pr, vp2.T))
	eq3 = np.dot(vp3, np.dot(pr, vp3.T))

	res = solve((eq1[0][0],eq2[0][0],eq3[0][0]), x,u,v)
	return res

def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.array([x, y, 1])[np.newaxis]

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	# print "entered"
	global refPt, cropping
 
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		# refPt = [(x, y)]
		refPt.append((x, y))
		cropping = True
		cv2.imshow("image", image)


def main():		
	global image
	image = cv2.imread("/home/chandu/CV-Codes/images/img1.jpg")
	# image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# nimg = image
	image = cv2.resize(image, (1024, 623))
	clone = image.copy()
	cv2.namedWindow("image")
	cv2.setMouseCallback("image", click_and_crop)
	# keep looping until the 'q' key is pressed
	while True:
		# display the image and wait for a keypress
		cv2.imshow("image", image)
		key = cv2.waitKey(1) & 0xFF
	 
		# if the 'r' key is pressed, reset the cropping region
		if key == ord("r"):
			image = clone.copy()
	 
		# if the 'c' key is pressed, break from the loop
		elif key == ord("c"):
			break
	 
	# if there are two reference points, then crop the region of interest
	# from the image and display it
	print(refPt)
	
	#line1
	A = [refPt[0][0],refPt[0][1]]
	B = [refPt[1][0],refPt[1][1]]
	cv2.line(image,(refPt[0][0],refPt[0][1]),(refPt[1][0],refPt[1][1]),(255, 0, 0), 3)
	

	#line2
	C = [refPt[2][0],refPt[2][1]]
	D = [refPt[3][0],refPt[3][1]]
	cv2.line(image,(refPt[2][0],refPt[2][1]),(refPt[3][0],refPt[3][1]),(255, 0, 0), 3)
	
	#line3
	E = [refPt[4][0],refPt[4][1]]
	F = [refPt[5][0],refPt[5][1]]
	cv2.line(image,(refPt[4][0],refPt[4][1]),(refPt[5][0],refPt[5][1]),(0, 255, 0), 3)
	
	#line4
	G = [refPt[6][0],refPt[6][1]]
	H = [refPt[7][0],refPt[7][1]]
	cv2.line(image,(refPt[6][0],refPt[6][1]),(refPt[7][0],refPt[7][1]),(0, 255, 0), 3)
	
	#line5
	I = [refPt[8][0],refPt[8][1]]
	J = [refPt[9][0],refPt[9][1]]
	cv2.line(image,(refPt[8][0],refPt[8][1]),(refPt[9][0],refPt[9][1]),(0, 0, 255), 3)
	
	#line6
	K = [refPt[10][0],refPt[10][1]]
	L = [refPt[11][0],refPt[11][1]]
	cv2.line(image,(refPt[10][0],refPt[10][1]),(refPt[11][0],refPt[11][1]),(0, 0, 255), 3)

	
	#cv2.imshow("image", image)
	vp1 = line_intersection((A,B), (C,D))
	vp2 = line_intersection((E,F), (G,H))
	vp3 = line_intersection((I,J), (K,L))
	
	w = len(image)
	h = len(image[0])

	#increasing image size in order to plot vps
	row1=row2=row3=0
	if vp1[0][0] < 0:
		row1 = abs(vp1[0][0])
		# vp1[0][0] = w - vp1[0][0]

	if vp2[0][0] < 0:
		row2 = abs(vp2[0][0])
		# vp2[0][0] = w - vp2[0][0]

	if vp2[0][0] < 0:
		row3 = abs(vp3[0][0])
		# vp3[0][0] = w - vp3[0][0]

	rows = max(row1, row2, row3)

	col1=col2=col3=0
	if vp1[0][1] < 0:
		col1 = abs(vp1[0][1])
		# vp2[0][0] = h - 

	if vp2[0][1] < 0:
		col2 = abs(vp2[0][1])
	if vp2[0][1] < 0:
		col3 = abs(vp3[0][1])
	cols = max(col1, col2, col3)
	
	# Changing row and column of vps if it is negative
	if vp1[0][0] < 0:
		vp1[0][0] = rows - abs(vp1[0][0])

	if vp2[0][0] < 0:
		vp2[0][0] = rows - abs(vp2[0][0])

	if vp2[0][0] < 0:
		vp3[0][0] = rows - abs(vp3[0][0])


	col1=col2=col3=0
	if vp1[0][1] < 0:
		vp3[0][0] = cols - abs(vp1[0][1])

	if vp2[0][1] < 0:
		vp3[0][0] = cols - abs(vp2[0][1])

	if vp2[0][1] < 0:
		vp3[0][0] = cols - abs(vp3[0][1])

	
	print(len(image), len(image[0]))
	new_image = np.full((rows+w, cols+h,3),255)
	# cv2.imshow("new_image", new_image)
	print(len(new_image), len(new_image[0]))
	# new_image = 255
	# for i in range(len(image)):
	# 	for j in range(len(image[0])):
	# 		new_image[rows+i, cols+j] = image[i,j]
	new_image[rows:rows+w, cols:cols+h] = image
	#--------------------------------------------------------------


	print (vp1, vp2, vp3)

	#plotting vanishing points on image
	cv2.circle(new_image,(vp1[0][0], vp1[0][1]), 5, (0,255,0), -1)
	cv2.line(new_image,(B[0],B[1]),(vp1[0][0],vp1[0][1]),(255, 0, 0), 3)
	cv2.line(new_image,(D[0],D[1]),(vp1[0][0],vp1[0][1]),(255, 0, 0), 3)

	cv2.circle(new_image,(vp2[0][0], vp2[0][1]), 5, (0,0,255), -1)
	cv2.line(new_image,(F[0],F[1]),(vp2[0][0],vp2[0][1]),( 0, 255, 0), 3)
	cv2.line(new_image,(H[0],H[1]),(vp2[0][0],vp2[0][1]),( 0, 255, 0), 3)

	cv2.circle(new_image,(vp3[0][0], vp3[0][1]), 5, (255,0,0), -1)
	cv2.line(new_image,(J[0],J[1]),(vp3[0][0],vp3[0][1]),(0, 0, 255), 3)
	cv2.line(new_image,(L[0],L[1]),(vp3[0][0],vp3[0][1]),(0, 0, 255), 3)

	#--------------------------------------------------------------
	image = cv2.resize(image, (1024, 623))
	new_image = new_image.astype(np.uint8)
	new_image = cv2.resize(new_image, (1024, 623))
	
	cv2.imshow("new_image", new_image)

	cv2.waitKey(0)
	

	# close all open windows
	cv2.destroyAllWindows()

	#Finding camera intrinsic parameters f,u,v
	res = find_camParams(vp1, vp2, vp3)
	print ("res:", res)
	#-----------------------------------------
if __name__=="__main__":
	main()	