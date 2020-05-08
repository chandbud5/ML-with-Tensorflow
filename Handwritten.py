import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def predict():
    n = int(input("Please enter a image number between 0-10,000 "))
    # Getting Dataset
    mnist = tf.keras.datasets.mnist
    (train_x, train_y), (test_x, test_y) = mnist.load_data()

    # Callbacks
    class MyCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('accuracy') > 0.99):
                print("\nStopping training as model became 99% accurate")
                self.model.stop_training = True

    callme = MyCallback()

    # Scaling X(Pixel values)
    train_x, test_x = train_x / 255.0, test_x / 255.0

    # Creating neural network
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(train_x, train_y, epochs=10, callbacks=[callme])

    plt.imshow(test_x[n])
    plt.show()

    print("IMAGE is shown on your right")

    ex = test_x[n]
    ex.resize((1, 28, 28))
    prediction = model.predict(ex)

    print("\nAccording to neural network it seems to be digit ", prediction.argmax())


predict()