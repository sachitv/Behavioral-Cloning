import cv2
import numpy as np
import math

def WhiteOutSides(img):
	whitened = img
	
	width = whitened.shape[1]
	height = whitened.shape[0]
	halfwidth = int(width/2)
	
	for i in range(height):
		for j in range(halfwidth,width):
			if(img[i][j] == 255):
				img[i:i+1, j:width] = 255
				break
		for j in range(0,halfwidth):
			iter = halfwidth - j
			if(img[i][iter] == 255):
				img[i:i+1, 0:iter] = 255
				break				
	return whitened

def DrawLines(img, lines, color=[255, 255, 255], thickness=2):
	for line in lines:
	    for x1,y1,x2,y2 in line:
	        cv2.line(img, (x1,y1), (x2,y2), color, thickness)


def PreprocessImage(img):
	cropped = img[55:136, 0:320] # Crop from x, y, w, h -> 0, 55, 320, 136
	resized_image = cv2.resize(cropped, (200, 66))
	hsv = cv2.cvtColor(resized_image,cv2.COLOR_BGR2HSV)
	
	#Enhance Lines and Hide other features
	rho = 1; #pixel
	theta = math.pi / 180;
	threshold = 20;
	min_line_len = 12;
	max_line_gap = 10;
	
	kernel_size = 5
	
	gaussian = cv2.GaussianBlur(hsv, (kernel_size, kernel_size), 0)	
	
	#use the saturation value
	grayscaleImage = cv2.split(hsv)[1]
	cannyImg = cv2.Canny(grayscaleImage, 50, 150)
	
	lines = cv2.HoughLinesP(cannyImg, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
			
	mask3Channel = np.zeros_like(gaussian)
	DrawLines(mask3Channel, lines)
	mask = cv2.cvtColor(mask3Channel,cv2.COLOR_RGB2GRAY)
	#mask = WhiteOutSides(mask)
    #cv2.imwrite('mask.jpg', mask)
	
	invertedMask = cv2.bitwise_not(mask)
	
	maskedImage = cv2.bitwise_and(gaussian,gaussian,mask = invertedMask)
	
	weightedImage = cv2.addWeighted(gaussian, 0.5, maskedImage, 0.5, 0)
	
	return weightedImage
	
def main():
	image = cv2.imread("output/2017_02_13_03_50_27_850.jpg")
	#image = cv2.imread("workingtraining_data/IMG/center_2017_02_12_08_24_51_858.jpg")
	image = PreprocessImage(image)	
	cv2.imwrite('temp.jpg', image)
	
#main()
