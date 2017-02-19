#imports
import os
import csv
import cv2
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Activation
from keras.layers.convolutional import Convolution2D, Cropping2D
from common import PreprocessImageTrain, PreprocessImageTest

def CreateModel():
    #some defines
    ch, row, col = 3, 66, 200  # Image format
    
    # set up Model layers
    model = Sequential()
    
    model.add(Lambda(lambda x: ((x / 255.0) - 0.5), input_shape=(row,col,ch)))
    
    model.add(Convolution2D(8, 5, 5))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Convolution2D(16, 4, 4))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dropout(0.3))
    model.add(Activation('relu'))
    
    model.add(Dense(128))
    model.add(Dropout(0.5))
    model.add(Activation('relu'))
    
    model.add(Dense(10))
    model.add(Activation('relu'))
    
    model.add(Dense(1))
    
    return model



def training_generator(samples, num_images_at_once=32, num_augmentations = 8):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, num_images_at_once):
            batch_samples = samples[offset:offset+num_images_at_once]
            
            images_per_generation = num_images_at_once * num_augmentations * 3 # 3 because of left, right, center
            
            images = np.empty((images_per_generation, 66, 200, 3))
            angles = np.empty((images_per_generation))
                        
            start = 0
                        
            for batch_sample in batch_samples:
            	end = start + num_augmentations
            	
            	#center image
            	name = 'training_data/IMG/'+batch_sample[0].split('/')[-1]
            	center_image = cv2.imread(name)
            	center_angle = float(batch_sample[3])
            	images[start:end,:,:,:], angles[start:end] = PreprocessImageTrain(center_image, num_augmentations, center_angle)
		
            	start = end
            	end = start+num_augmentations
            	
            	#left image
            	name = 'training_data/IMG/'+batch_sample[1].split('/')[-1]
            	left_image = cv2.imread(name)
            	left_angle = float(batch_sample[3]) + 0.25
            	images[start:end,:,:,:], angles[start:end] = PreprocessImageTrain(left_image, num_augmentations, left_angle)
            	
            	start = end
            	end = start+num_augmentations
            	
            	#right image
            	name = 'training_data/IMG/'+batch_sample[2].split('/')[-1]
            	right_image = cv2.imread(name)
            	right_angle = float(batch_sample[3]) - 0.25            	
            	images[start:end,:,:,:], angles[start:end] = PreprocessImageTrain(right_image, num_augmentations, right_angle)
            	            	
            	start = end
            
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield shuffle(X_train, y_train)
            
            
            
def validation_generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = 'training_data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_image = PreprocessImageTest(center_image)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
            
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield shuffle(X_train, y_train)



def main():
    samples = []
    with open('training_data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    
    from sklearn.model_selection import train_test_split
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)

    numEpochs = 10
    numAugmentations = 8
    
    num_images_at_once = 5
    training_batch_size = len(train_samples) * numAugmentations * 3  # for 3 images- left, right, center.
    validation_batch_size = 8
    training_samples_per_epoch = training_batch_size
    
    
    # compile and train the model using the generator function
    train_generator = training_generator(train_samples, num_images_at_once=num_images_at_once, num_augmentations=numAugmentations)
    val_generator = validation_generator(validation_samples, batch_size=validation_batch_size)
    
    model = CreateModel();
    
    # Preprocess incoming data, centered around zero with small standard deviation
    model.compile(loss='mse', optimizer='adam')
    model.fit_generator(train_generator, samples_per_epoch=training_samples_per_epoch, validation_data=val_generator, nb_val_samples=len(validation_samples), nb_epoch=numEpochs)
    model.save('model.h5')

main()
