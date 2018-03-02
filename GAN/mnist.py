import keras
import numpy
import sklearn
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Conv2D, Dense, Dropout, Flatten, BatchNormalization, Reshape, Conv2DTranspose, UpSampling2D, Activation, GlobalAveragePooling2D, Input, ZeroPadding2D
from numpy.random import random
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
import pathlib
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
log = logging.getLogger()

(x_train, y_train), (x_test, y_test) = mnist.load_data()
MNIST = numpy.concatenate((x_train, x_test)) / 255.  #length is 70000

DEPTH = 64
KERNEL = 5
STRIDES = 2
PADDING = 'same'
ACTIVATION = 'elu'
DROPOUT = 0.4
BATCH_NORM = 0.9
PATIENCE = 2
SAMPLE_SIZE = 10000
SEED = 42


def d_conv_layer(inputs, multiplier, stride=STRIDES, trainable=True):
    layer = Conv2D(
        name='d_conv_%s' % multiplier,
        strides=stride,
        filters=DEPTH * 1,
        kernel_size=KERNEL,
        input_shape=inputs.shape,
        padding=PADDING,
        activation=ACTIVATION,
        trainable=trainable)(inputs)
    layer = Dropout(DROPOUT)(layer)
    return layer


def discriminator_layers(inputs, trainable=True):
    layers = d_conv_layer(inputs, 1, trainable=trainable)
    layers = d_conv_layer(layers, 2, trainable=trainable)
    layers = d_conv_layer(layers, 4, trainable=trainable)
    layers = d_conv_layer(layers, 8, 1, trainable=trainable)
    layers = Flatten()(layers)
    layers = Dense(
        1, name="d_output", activation='sigmoid', trainable=trainable)(layers)
    return layers


def g_conv_layer(inputs, multiplier, trainable=True):
    x = Conv2DTranspose(
        DEPTH * multiplier,
        KERNEL,
        name='g_conv_%s' % multiplier,
        padding=PADDING,
        trainable=trainable)(inputs)
    x = BatchNormalization(momentum=BATCH_NORM)(x)
    x = Activation(ACTIVATION)(x)
    x = UpSampling2D()(x)
    return x


def generator_layers(inputs, trainable=True):
    size = 3
    x = Dense(
        DEPTH * 8 * size * size,
        name="g_dense",
        input_shape=inputs.shape,
        trainable=trainable)(inputs)
    x = BatchNormalization(momentum=BATCH_NORM)(x)
    x = Activation(ACTIVATION)(x)
    x = Reshape((size, size, DEPTH * 8))(x)
    x = Dropout(DROPOUT)(x)
    x = g_conv_layer(x, 4)
    x = g_conv_layer(x, 2)
    x = ZeroPadding2D((1, 1))(x)
    x = g_conv_layer(x, 1)
    x = Conv2DTranspose(
        1, KERNEL, name='g_output', padding=PADDING, trainable=trainable)(x)
    x = Activation('sigmoid')(x)
    return x


def train_generator():
    gen_inputs = Input(shape=(100, ))
    gen_layers = generator_layers(gen_inputs, trainable=True)
    disc_layers = discriminator_layers(gen_layers, trainable=True)
    adversarial = Model(inputs=gen_inputs, outputs=disc_layers)
    adversarial.compile(
        optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    adversarial.summary()

    if pathlib.Path("./adversarial.hdf5").is_file():
        adversarial.load_weights("adversarial.hdf5", by_name=True)

    inputs = random((SAMPLES, 100))
    labels = numpy.ones(SAMPLES)

    X_train, X_test, y_train, y_test = train_test_split(
        inputs, labels, test_size=0.2, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED)

    checkpointer = keras.callbacks.ModelCheckpoint(
        "adversarial.hdf5",
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        period=1)
    stopper = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=PATIENCE, min_delta=.02, verbose=1)

    adversarial.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        verbose=1,
        callbacks=[checkpointer, stopper],
        validation_data=(X_val, y_val))


def show_generated_images():
    gen_inputs = Input(shape=(100, ))
    gen_layers = generator_layers(gen_inputs, trainable=False)
    generator = Model(inputs=gen_inputs, outputs=gen_layers)
    generator.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    if pathlib.Path("./adversarial.hdf5").is_file():
        generator.load_weights("adversarial.hdf5", by_name=True)

    frames = random((5, 100))
    fakes = generator.predict(frames, verbose=1)
    fakes = numpy.squeeze(fakes)

    reals = MNIST[5:, :, :]

    w = 10
    h = 10
    fig = plt.figure(figsize=(8, 8))
    rows = 2
    cols = 6
    for i in range(1, cols):
        ax = fig.add_subplot(rows, cols, i)
        plt.imshow(fakes[i - 1])
    for i in range(1, cols):
        ax = fig.add_subplot(rows, cols, i + cols)
        plt.imshow(reals[i - 1])
    plt.show()


def generator_model(trainable):
    log.info("MAKING GENERATOR MODEL")
    gen_inputs = Input(shape=(100, ))
    gen_layers = generator_layers(gen_inputs, trainable=trainable)
    generator = Model(inputs=gen_inputs, outputs=gen_layers)
    generator.compile(
        optimizer='rmsprop',
        loss='categorical_crossentropy',
        metrics=['accuracy'])
    if pathlib.Path("./adversarial.hdf5").is_file():
        log.info("LOADING GENERATOR WEIGHTS")
        generator.load_weights("adversarial.hdf5", by_name=True)
    return generator


def generate_fakes(quantity):
    generator = generator_model(False)
    noise = random((quantity, 100))
    log.info("GENERATING FAKES")
    fakes = generator.predict(noise, verbose=1)
    return fakes


def get_reals(quantity):
    reals = MNIST[:quantity, :, :]
    reals = numpy.expand_dims(reals, axis=3)
    return reals    


def get_data_sets(sample_size):
    """returns training, test, and validation sets"""
    log.info("MAKING DATA SETS")
    training_bound = int(sample_size * 0.7)
    test_bound = int(sample_size * 0.8)

    fakes = generate_fakes(int(sample_size / 2))
    reals = get_reals(int(sample_size / 2))
    total = numpy.concatenate((reals, fakes))
    real_labels = numpy.ones([len(reals)])
    fake_labels = numpy.zeros([len(fakes)])
    total_labels = numpy.concatenate((real_labels, fake_labels))
    shuffled, shuffled_labels = shuffle(total, total_labels)

    X_train = shuffled[:training_bound, :, :]
    X_val = shuffled[training_bound:test_bound, :, :]
    X_test = shuffled[test_bound:, :, :]

    y_train = shuffled_labels[:training_bound]
    y_val = shuffled_labels[training_bound:test_bound]
    y_test = shuffled_labels[test_bound:]

    return {
        "train": [X_train, y_train],
        "test": [X_test, y_test],
        "val": [X_val, y_val]
    }


def discriminator_model(trainable):
    log.info("MAKING DISCRIMINATOR MODEL")
    disc_inputs = Input(shape=(28, 28, 1))
    disc_layers = discriminator_layers(disc_inputs, trainable=trainable)
    discriminator = Model(inputs=disc_inputs, outputs=disc_layers)
    discriminator.compile(
        optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    discriminator.summary()
    if pathlib.Path("./discriminator.hdf5").is_file():
        log.info("LOADING DISCRIMINATOR WEIGHTS")
        discriminator.load_weights("discriminator.hdf5", by_name=True)
    return discriminator


def train_discriminator():
    data_sets = get_data_sets(SAMPLE_SIZE)
    discriminator = discriminator_model(True)
    checkpointer = keras.callbacks.ModelCheckpoint(
        "discriminator.hdf5",
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode='auto',
        period=1)
    stopper = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=PATIENCE, verbose=1)
    discriminator.fit(
        data_sets["train"][0],
        data_sets["train"][1],
        epochs=10,
        batch_size=32,
        verbose=1,
        callbacks=[checkpointer, stopper],
        validation_data=(data_sets["val"][0], data_sets["val"][1]))

    scores = discriminator.evaluate(data_sets["test"][0], data_sets["test"][1])
    log.info("METRICS: %s %s" % (discriminator.metrics_names, scores))


def show_classified_images():
    fakes = generate_fakes(5)
    reals = get_reals(5)
    total = numpy.concatenate((fakes, reals))
    total = numpy.squeeze(total)
    total = numpy.expand_dims(total, axis=3)

    shuffle(total)

    discriminator = discriminator_model(False)
    classes = discriminator.predict(total, verbose=1)

    total = numpy.squeeze(total)

    w = 10
    h = 10
    fig = plt.figure(figsize=(8, 8))
    rows = 2
    cols = 5
    for i in range(1, rows * cols):
        ax = fig.add_subplot(rows, cols, i)
        ax.set_title("Class: %i" % classes[i])
        plt.imshow(total[i])
    plt.show()


#train_discriminator()
show_classified_images()
