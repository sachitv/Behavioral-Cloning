import cv2
import numpy as np
import math

def alterBrightness(img):
	img = ConvertToHSV(img)
	random_bright = .25+np.random.uniform()
	img[:,:,2] = img[:,:,2]*random_bright
	return img
	
def trans_image(image,steer):
    # Translation
    trans_range_x = 40
    trans_range_y = 10
    rows,cols,channels = image.shape
    
    tr_x = trans_range_x*np.random.uniform()-trans_range_x/2
    steer_ang = steer + tr_x/trans_range_x*2*0.2
    
    tr_y = trans_range_y*np.random.uniform()-trans_range_y/2
    Trans_M = np.float32([[1,0,tr_x],[0,1,tr_y]])
    image_tr = cv2.warpAffine(image,Trans_M,(cols,rows), borderMode=cv2.BORDER_REPLICATE)
    
    return image_tr,steer_ang

def CropResizeImage(img):
	cropped = img[55:136, 0:320] # Crop from x, y, w, h -> 0, 55, 320, 136
	resized = cv2.resize(cropped, (200, 66))
	return resized

def ConvertToHSV(img):
	hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
	return hsv
	
def PreprocessImageTrain(img, combinations_per_augmentation, inputsteer = 0):
	#Generate combination Images
	combinations = []
	steeringAngles = []
	
	#add the original separately
	combinations.append( CropResizeImage( ConvertToHSV(img ) ) )
	steeringAngles.append(inputsteer)
	
	#make some augmentations
	for i in range(combinations_per_augmentation - 1):
		newImage = alterBrightness(img)
		newImage = CropResizeImage(newImage)
		newImage, newSteer = trans_image(newImage, inputsteer)
		combinations.append(newImage)
		steeringAngles.append(newSteer)
	
	combinations = np.array(combinations)
	steeringAngles = np.array(steeringAngles)
	
	return combinations, steeringAngles
	
def PreprocessImageTest(img):
	resized = CropResizeImage(img)
	hsv = ConvertToHSV(resized)
	return hsv
	
def test():
	image = cv2.imread("training_data/IMG/center_2017_02_12_08_24_32_131.jpg")
	
	images = np.empty((10, 66, 200, 3))
	steering = np.empty((10))
		
	images[0:10,:,:,:], steering[0:10] = PreprocessImageTrain(image, 10, 5.0)
			
	for i in range(len(images)):
		cv2.imwrite("temp" + str(i) + "-steer:"+str(steering[i])+".jpg", images[i])
	
#test()
