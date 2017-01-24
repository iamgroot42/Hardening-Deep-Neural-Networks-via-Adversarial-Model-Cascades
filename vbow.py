import cv2
import numpy as np
from sklearn.cluster import KMeans
import utils_cifar


def sift_vector(x):
	gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	kp, desc = sift.detectAndCompute(gray, None)
	return desc


def gen_sift_features(X):
	img_descs = []
	for data_point in X:
		img_descs.append(sift_vector(desc))
	img_descs = np.array(img_descs)
	return img_descs


def cluster_features(img_descs, cluster_model):
	# Concatenate all descriptors in the training set together
	all_train_descriptors = []
	for desc_list in img_descs:
		try:
			for desc in desc_list:
				all_train_descriptors.append(desc)
		except:
			continue
	all_train_descriptors = np.array(all_train_descriptors)
	cluster_model.fit(all_train_descriptors)
	img_clustered_words = [cluster_model.predict(raw_words) for raw_words in img_descs]
	img_bow_hist = np.array([np.bincount(clustered_words, minlength=cluster_model.n_clusters) for clustered_words in img_clustered_words])
	return img_bow_hist


def img_to_vect(data_point, cluster_model):
	desc = sift_vector(data_point)
	clustered_desc = cluster_model.predict(desc)
	img_bow_hist = np.bincount(clustered_desc, minlength=cluster_model.n_clusters)
	print img_bow_hist


if __name__ == "__main__":
	(X_train, Y_train, X_test, Y_test) = utils_cifar.data_cifar_raw()
	transformed_train = gen_sift_features(X_train)
	kmeans = KMeans(n_clusters=10, random_state=0)
	cluster_features(transformed_train[:100], kmeans)
	print img_to_vect(X_test[0], kmeans)
