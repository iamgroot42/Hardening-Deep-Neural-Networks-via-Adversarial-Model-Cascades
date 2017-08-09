import sys
import cv2
import numpy as np


def scaleImages(images, x, y):
	rescaled_images = np.copy(images)
	rescaled_images = np.transpose(rescaled_images, (0,2,3,1)) * 255
	rescaled_images = rescaled_images.astype('uint8')
	new_images = []
	for image in rescaled_images:
		new_images.append(cv2.resize(image, (x,y), interpolation=cv2.INTER_CUBIC))
	return np.transpose(np.array(new_images), (0,3,1,2)).astype('float32') / 255.0


def shiftImages(images, x, y, shift_x, shitf_y):
	rescaled_images = np.copy(images)
        rescaled_images = np.transpose(rescaled_images, (0,2,3,1)) * 255
        rescaled_images = rescaled_images.astype('uint8')
        new_images = []
	M = np.float32([[1,0,shift_x],[0,1,shift_y]])
        for image in rescaled_images:
                new_images.append(cv2.warpAffine(image,M,(x,y)))
        return np.transpose(np.array(new_images), (0,3,1,2)).astype('float32') / 255.0


def rotateImages(images, x, y, angle):
	rescaled_images = np.copy(images)
        rescaled_images = np.transpose(rescaled_images, (0,2,3,1)) * 255
        rescaled_images = rescaled_images.astype('uint8')
        new_images = []
	M = cv2.getRotationMatrix2D((x/2,y/2),angle,1)
        for image in rescaled_images:
                new_images.append(cv2.warpAffine(image,M,(x,y)))
        return np.transpose(np.array(new_images), (0,3,1,2)).astype('float32') / 255.0


if __name__ == "__main__":
	source = np.load(sys.argv[1])
	#np.save(sys.argv[1], scaleImages(source, 32, 32)) #Scale images
	np.save(sys.argv[1], rotateImages(source, 32, 32, 3, 3)) #Rotate images
	#np.save(sys.argv[1], rotateImages(source, 32, 32, 90)) #Translate images
