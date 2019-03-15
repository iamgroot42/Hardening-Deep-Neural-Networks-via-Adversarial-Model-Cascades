from keras.layers import Conv2D, Dense, Activation, MaxPooling2D, Flatten
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Sequential
from keras import optimizers
from keras.regularizers import l2

def scheduler(epoch):
	if epoch < 60:
		return 0.05
	if epoch <= 120:
		return 0.01
	if epoch <= 160:
		return 0.002
	return 0.0004

def lenet_network(n_classes=10, is_mnist=False):
	weight_decay = 0.0001
	model = Sequential()
	input_shape=(32,32,3)
	if is_mnist:
		input_shape=(28,28,1)
	model.add(Conv2D(6, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay), input_shape=input_shape))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(Conv2D(16, (5, 5), padding='valid', activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay)))
	model.add(MaxPooling2D((2, 2), strides=(2, 2)))
	model.add(Flatten())
	model.add(Dense(120, activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay) ))
	model.add(Dense(84, activation = 'relu', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay) ))
	model.add(Dense(10, activation = 'softmax', kernel_initializer='he_normal', kernel_regularizer=l2(weight_decay) ))
	sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
	model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
	cbks = [LearningRateScheduler(scheduler),
		ModelCheckpoint('./checkpoint-{epoch}.h5', save_best_only=False, mode='auto', period=10)]
	return model, cbks

if __name__ == "__main__":
	model = lenet_network()