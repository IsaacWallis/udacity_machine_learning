import keras
import numpy
import sklearn
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten
from numpy.random import random
from sklearn.model_selection import train_test_split

(x_train, y_train), (x_test, y_test) = mnist.load_data()
real = numpy.concatenate((x_train, x_test))
real = real[:5000,:,:]
print (len(real))
fake = random((len(real), 28, 28))
total = numpy.concatenate((real, fake))
total = numpy.expand_dims(total, axis=3)
total = total / 255.

real_labels = numpy.ones((len(real)))
fake_labels = numpy.zeros((len(real)))
total_labels = numpy.concatenate((real_labels, fake_labels))

X_train, X_test, y_train, y_test = train_test_split(total, total_labels, test_size=0.2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2)

depth = 64
kernel = 5
strides = 2
padding = 'same'
activation = 'relu'
dropout = 0.4

discriminator = Sequential()
discriminator.add(Conv2D(filters=depth*1, kernel_size=kernel, strides=strides, input_shape=(28, 28, 1), padding=padding, activation=activation))
discriminator.add(Dropout(dropout))
discriminator.add(Conv2D(filters=depth*2, kernel_size=kernel, strides=strides, padding=padding, activation=activation))
discriminator.add(Dropout(dropout))
discriminator.add(Conv2D(filters=depth*4, kernel_size=kernel, strides=strides, padding=padding, activation=activation))
discriminator.add(Dropout(dropout))
discriminator.add(Conv2D(filters=depth*8, kernel_size=kernel, strides=1, padding=padding, activation=activation))
discriminator.add(Dropout(dropout))
discriminator.add(Flatten())
discriminator.add(Dense(1, activation='sigmoid'))
discriminator.summary()

discriminator.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])


#checkpointer = keras.callbacks.ModelCheckpoint("discriminator.hdf5", monitor='val_loss', save_best_only=True, save_weights_only=False, mode='auto', period=1)
#discriminator.fit(X_train, y_train, epochs=1, batch_size=32, verbose=1, callbacks=[checkpointer], validation_data=(X_val, y_val))


discriminator.load_weights("discriminator.hdf5")
predictions = numpy.array([discriminator.predict(numpy.expand_dims(tensor,axis=0))[0,0] for tensor in X_test], dtype='int32')
accuracy = 100 * numpy.sum(predictions==numpy.array(y_test)) / len(y_test)

print(accuracy)