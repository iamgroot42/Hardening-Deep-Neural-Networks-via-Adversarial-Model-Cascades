import cv2
import numpy as np
import utils_cifar


def sift_vector(x):
	gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype('uint8')
	sift = cv2.xfeatures2d.SIFT_create()
	kp, desc = sift.detectAndCompute(gray.astype('uint8'), None)
	return desc


def gen_sift_features(X):
	img_descs = []
	for data_point in X:
		img_descs.append(sift_vector(data_point))
	return np.array(img_descs)


def cluster_features(img_descs, cluster_model):
	img_descs = gen_sift_features(img_descs)
	all_train_descriptors = []
	img_descs_temp = []
	for raw_words in img_descs:
		if raw_words is None:
			continue
		img_descs_temp.append(raw_words)
	for desc_list in img_descs_temp:
		for desc in desc_list:
			all_train_descriptors.append(desc)
	all_train_descriptors = np.array(all_train_descriptors)
	cluster_model.fit(all_train_descriptors)
	img_clustered_words = [cluster_model.predict(raw_words) for raw_words in img_descs_temp]
	img_bow_hist = np.array([np.bincount(clustered_words, minlength=cluster_model.n_clusters) for clustered_words in img_clustered_words])
	return img_bow_hist, cluster_model


def img_to_vect(X, cluster_model):
	img_descs = []
	for data_point in X:
		desc = sift_vector(data_point)
		if desc is None:
			clustered_desc = np.zeros((cluster_model.n_clusters,)).astype('int64')
		else:
			clustered_desc = cluster_model.predict(desc)
		img_bow_hist = np.bincount(clustered_desc, minlength=cluster_model.n_clusters)
		img_descs.append(img_bow_hist)
	return np.array(img_descs).astype('float64')
