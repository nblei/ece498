import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow import keras

def make_model():
    model = keras.Sequential([
        keras.layers.Conv2D(input_shape=[28,28,1], filters=3, kernel_size=[7, 7], use_bias=True,
                           activation='relu', padding='SAME', dilation_rate=2),
        keras.layers.MaxPool2D(pool_size=[2, 2]),
        keras.layers.SeparableConv2D(filters=6, kernel_size=[3, 3], use_bias=True,
                                    padding='same', activation='relu'),
        keras.layers.MaxPool2D(pool_size=[2, 2]),
        keras.layers.Flatten(),
        keras.layers.Dense(100, activation='relu', use_bias=True),
        keras.layers.Dense(50, activation='relu', use_bias=True),
        keras.layers.Dense(10, activation='softmax', use_bias=True)
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

    return model

def main():
    NUM_EPOCHS = 50
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

