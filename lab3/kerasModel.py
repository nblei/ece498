#import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras

def make_sequential_model():
    model = keras.Sequential([
        keras.layers.Conv2D(input_shape=[28,28,1], filters=3, kernel_size=[7, 7], use_bias=True,
                           activation='relu', padding='SAME', dilation_rate=3),
        keras.layers.MaxPool2D(pool_size=[2, 2]),
        keras.layers.SeparableConv2D(filters=6, kernel_size=[3, 3], use_bias=True,
                                    padding='same', activation='relu'),
        keras.layers.MaxPool2D(pool_size=[2, 2]),
        keras.layers.Flatten(),
        keras.layers.Dense(100, activation='relu', use_bias=True),
        keras.layers.Dense(50, activation='relu', use_bias=True),
        keras.layers.Dense(10, activation='softmax', use_bias=True)
    ])
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    return model

def make_sequential_dwise_model():
    model = keras.Sequential([
        keras.layers.Conv2D(input_shape=[28,28,1], filters=3, kernel_size=[7, 7], use_bias=True,
                           activation='relu', padding='SAME', dilation_rate=3),
        keras.layers.MaxPool2D(pool_size=[2, 2]),
        keras.layers.SeparableConv2D(filters=6, kernel_size=[3, 3], use_bias=True,
                                    padding='same', activation='relu'),
        keras.layers.MaxPool2D(pool_size=[2, 2]),
        keras.layers.Flatten(),
        keras.layers.Dense(100, activation='relu', use_bias=True),
        keras.layers.Dense(50, activation='relu', use_bias=True),
        keras.layers.Dense(10, activation='softmax', use_bias=True)
    ])
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    return model


def make_only_dense():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=[28,28,1]),
        #keras.layers.BatchNormalization(),
        keras.layers.Dense(28*28, activation='relu',use_bias=False),
        #keras.layers.BatchNormalization(),
        keras.layers.Dense(28*28, activation='relu',use_bias=False),
        #keras.layers.BatchNormalization(),
        keras.layers.Dense(50, activation='relu',use_bias=False),
        #keras.layers.BatchNormalization(),
        keras.layers.Dense(10, activation='softmax')
    ])
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    return model

def make_inception_model():
    NUM_FEATURES=3
    input_img = keras.layers.Input(shape=(28, 28, 1))
    tower_1 = keras.layers.Conv2D(NUM_FEATURES, (1, 1), padding='same', activation='relu')(input_img)
    tower_1 = keras.layers.Conv2D(NUM_FEATURES, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = keras.layers.Conv2D(NUM_FEATURES, (1,1), padding='same', activation='relu')(input_img)
    tower_2 = keras.layers.Conv2D(NUM_FEATURES, (5,5), padding='same', activation='relu')(tower_2)

    tower_3 = keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
    tower_3 = keras.layers.Conv2D(NUM_FEATURES, (1,1), padding='same', activation='relu')(tower_3)

    output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)
    output = keras.layers.Flatten()(output)

    out = keras.layers.Dense(10, activation='softmax')(output)
    model = keras.models.Model(inputs=input_img, outputs=out)
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model

def make_xception_model():
    DROP_RATE = 0.3
    NUM_FEATURES=2
    input_img = keras.layers.Input(shape=(28, 28, 1))
    input_processing = keras.layers.Conv2D(NUM_FEATURES, (3, 3), padding='same', activation='relu')(input_img)
    input_processing = keras.layers.BatchNormalization()(input_processing)

    tower_1 = keras.layers.Conv2D(NUM_FEATURES, (1, 1), padding='same', activation='relu')(input_processing)
    tower_1 = keras.layers.SeparableConv2D(NUM_FEATURES, (3, 3), padding='same', activation='relu')(tower_1)

    tower_2 = keras.layers.Conv2D(NUM_FEATURES, (1,1), padding='same', activation='relu')(input_processing)
    tower_2 = keras.layers.SeparableConv2D(NUM_FEATURES, (5,5), padding='same', activation='relu')(tower_2)

    tower_3 = keras.layers.MaxPooling2D((3,3), strides=(1,1), padding='same')(input_processing)
    tower_3 = keras.layers.Conv2D(NUM_FEATURES, (1,1), padding='same', activation='relu')(tower_3)

    output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=3)
    output = keras.layers.Flatten()(output)

    #dense = keras.layers.Dense(100, activation='relu')(output)
    #dense = keras.layers.Dense(50, activation='relu')(output)

    out = keras.layers.Dense(10, activation='softmax')(output)
    model = keras.models.Model(inputs=input_img, outputs=out)
    opt = keras.optimizers.Adam()
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model

def make_model():
    return make_inception_model()


def main():
    NUM_EPOCHS = 5
    model = make_model()

    for layer in model.layers:
        print(layer.output_shape)


    #cp_callback = keras.callbacks.ModelCheckpoint("/home/nathan/school/ece498/mp2keras.save", period=5)

    fashion_mnist = keras.datasets.fashion_mnist

    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


    train_images = train_images / 255.0
    train_images.resize((60000, 28, 28, 1))
    test_images = test_images / 255.0
    test_images.resize((10000, 28, 28, 1))
    print(train_images.shape)

    model.fit(train_images, train_labels, epochs=NUM_EPOCHS)
    model.save_weights("./keras_model")

    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print("Test accuracy:", test_acc)

if __name__ == "__main__":
    print("Here!")
    main()

