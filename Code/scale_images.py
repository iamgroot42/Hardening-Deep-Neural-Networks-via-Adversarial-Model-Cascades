import sys
import cv2
import numpy as np


def scaleImages(images, scale):
	rescaled_images = np.copy(images)
	rescaled_images = np.transpose(rescaled_images, (0,2,3,1)) * 255
	rescaled_images = rescaled_images.astype('uint8')
	new_images = []
	for image in rescaled_images:
		new_images.append(cv2.resize(image, (0,0), fx=scale, fy=scale,  interpolation=cv2.INTER_CUBIC))
	return np.transpose(np.array(new_images), (0,3,1,2)).astype('float32') / 255.0


def shiftImages(images, x, y, shift_x, shift_y):
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
	scale = float(sys.argv[2])
	output = sys.argv[3]
	shift = int(sys.argv[2])
	angle = (360 + int(sys.argv[2])) % 360
	#np.save(sys.argv[3], scaleImages(source, scale)) #Scale images
	np.save(sys.argv[3], shiftImages(source, 32, 32, shift, shift)) #Translate images
        #np.save(sys.argv[3], rotateImages(source, 32, 32, angle)) #Rotate images

