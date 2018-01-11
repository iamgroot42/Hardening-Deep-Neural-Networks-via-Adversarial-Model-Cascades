import numpy as np
import os
import pickle


def get_label_mapping(filename):
    f = open(filename, 'r')
    mapping = {}
    for line in f:
        wnid, id, name = line.rstrip('\n').split(' ')
        mapping[wnid] = int(id)-1
    return mapping


def read_file(filename):
    f = open(filename, 'r')
    fids = []
    for line in f:
        fids.append(line.rstrip('\n'))
    return fids


def get_classes(mapping, required):
    mappings = []
    for req in required:
        mappings.append(mapping[req])
    return mappings


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict


def load_databatch(data_folder, idx, img_size=32):
    data_file = os.path.join(data_folder, 'train_data_batch_')

    d = unpickle(data_file + str(idx))
    x = d['data']
    y = d['labels']

    x = x/np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    data_size = x.shape[0]

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    # create mirrored images
    X_train = x[0:data_size, :, :, :]
    Y_train = y[0:data_size]
    X_train_flip = X_train[:, :, :, ::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train, X_train_flip), axis=0)
    Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

    return X_train.astype('float32'), Y_train.astype('int32')


if __name__ == "__main__":
    import sys
    label_mapping = get_label_mapping(sys.argv[1])
    required_images = read_file(sys.argv[2])
    req_classes = get_classes(label_mapping, required_images)
    relevant_X = []
    for j in range(10):
        X, Y = load_databatch(sys.argv[3], j+1)
        for i,x in enumerate(X):
            if Y[i] in req_classes:
                relevant_X.append(x)
    relevant_X = np.array(relevant_X)
    print(relevant_X.shape, "images extracted")
    np.save(sys.argv[4], relevant_X)

