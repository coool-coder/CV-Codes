import cv2
import numpy as np
import math


def main():
	img = cv2.imread("/home/chandu/CV-Codes/images/im52.png")
	img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	w = 3*len(img)
	h = 3*len(img[0])
	new_img = np.zeros((w, h))

	[tx,ty]  = map(int , raw_input("Enter translation parameter(tx,ty): ").split(","))
	# print(tx,ty,tz)
	# 3D_shift = np.array([[1,0,0,tx], [0,1,0,ty], [0,0,1,tz], [0,0,0,1]])
	shift = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

	# print(shift)
	# [Theta, Phi, Omega] = map(int, raw_input("Enter rotational parameter in degrees(theta, phi, omega):").split(","))
	Omega = input("Enter the amount of rotation (in degrees): ")
	# cosTheta = math.cos(math.radians(Theta))
	# sinTheta = math.sin(math.radians(Theta))

	# cosPhi = math.cos(math.radians(Phi))
	# sinPhi = math.sin(math.radians(Phi))

	cosOmega = math.cos(math.radians(Omega))
	sinOmega = math.sin(math.radians(Omega))


	# mat_about_Xaxis = np.array([[1, 0, 0], [0, cosTheta, -sinTheta], [0, sinTheta, cosTheta]])
	# mat_about_Yaxis = np.array([[cosPhi, 0, sinPhi], [0, 1, 0], [-sinPhi, 0, cosPhi]])
	mat_about_Zaxis = np.array([[cosOmega, -sinOmega, 0], [sinOmega, cosOmega, 0], [0, 0, 1]])
	
	#scaling
	[sx, sy] = map(float, raw_input("scaling factor (sx,sy): ").split(","))
	scale = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])

	#shearing 
	[shx, shy] = map(int, raw_input("scaling factor (shx,shy): ").split(","))
	shearing = np.array([[1, shx, 0], [shy, 1, 0], [0, 0, 1]])
	# Rotation_mat = np.dot(np.dot(mat_about_Xaxis, mat_about_Yaxis), mat_about_Zaxis)
	
	# Rotation = np.zeros((4,4))
	# Rotation[-1][-1] = 1
	# Rotation[0:3,0:3] = Rotation_mat

	# Projection = np.zeros((4,4))
	# Projection[0:3,0:3] = np.identity(3)

	# f = input("focal length: ")
	# alpha = input("aspect ratio: ")


	Trns_mat = np.dot(np.dot(np.dot(shearing, scale), mat_about_Zaxis), 	shift)
	print Trns_mat

	for x in range(len(img)):
		for y in range(len(img[0])):
			temp = np.array([x, y, 1])[np.newaxis]
			[nx, ny, w] = map(int, np.dot(Trns_mat, temp.T))
			new_img[nx+int(w/2), ny+int(h/2)] = img[x,y]

	cv2.imshow("img", img)

	new_img = new_img.astype(np.uint8)
	cv2.imshow("new_img", new_img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()


if __name__=="__main__":
	main()