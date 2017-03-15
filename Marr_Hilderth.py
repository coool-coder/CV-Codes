import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal as sg


img = cv2.imread("/home/chandu/Pictures/img6.jpg",0)

gauss = cv2.GaussianBlur(img,(5,5),0)
print type(gauss)

w,h = img.shape
print w , h

print "gauss:", '\n' , gauss

marr= cv2.Laplacian(gauss, cv2.CV_64F)	

marr= marr.astype(np.double)

# marr = sg.convolve(lap, gauss)
cv2.imshow("gaussian", gauss)

# marr = cv2.filter2D(lap, -1, g)
plt.hist(marr.ravel(),256,[0,256]); plt.show()
# if cv2.waitKey()==27:
# 	cv2.destroyAllWindows()

print "marr", '\n', marr	
thres = int(input())
# thres = 3
final = np.zeros(marr.shape)
o
for i in range(1,w-1):
	for j in range(1,h-1):
		if marr[i][j] > thres:
			#vertical
			if marr[i-1][j]*marr[i+1][j] < 0 and marr[i][j]>0: 
				final[i][j]=1
			#diagonal
			elif marr[i-1][j-1]*marr[i+1][j+1] < 0 and marr[i][j]>0:
				final[i][j]=1
			# horizontal
			elif marr[i][j-1]*marr[i][j+1] < 0 and marr[i][j]>0:
				final[i][j]=1

			elif marr[i-1][j+1]*marr[i+1][j-1] < 0 and marr[i][j]>0:
				final[i][j]=1

		# else:
		# 	final[i][j]=0

print "final", '\n', final
marr = marr.astype(np.uint8)
cv2.imshow("lapofgauss", marr)

# final = final.astype(np.uint8)
cv2.imshow("final", final)

cv2.waitKey(0)
cv2.destroyAllWindows()
