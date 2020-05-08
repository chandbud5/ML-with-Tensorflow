import tensorflow as tf
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt


# CALLBACKS
class myCall(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>=0.93):
            print("\nTraining stopped as accuracy 93% achieved")
            self.model.stop_training = True

mycallback = myCall()

# LOADING DATASET
train_x, train_y = loadlocal_mnist(images_path='train-images-idx3-ubyte',
                                   labels_path='train-labels-idx1-ubyte')

test_x, test_y = loadlocal_mnist(images_path='t10k-images-idx3-ubyte',
                                 labels_path='t10k-labels-idx1-ubyte')

# VISUALIZING DATASET
plt.imshow(train_x[110].reshape(28,28))
plt.title("Sample 110 image number")
plt.show()

# PREPROCESSING DATA
train_x = train_x.reshape(60000, 28, 28, 1)
test_x = test_x.reshape(10000, 28, 28, 1)
print(train_x.shape, test_x.shape, train_y.shape)


train_x, test_x = train_x/255.0, test_x/255.0


# CREATING MODEL OF CONVOLUTION NEURAL NETWORK
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu, input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256,activation = tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# COMPILING MODEL
model.compile(
    optimizer='adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.summary()

# FITTING THE MODEL
model.fit(train_x, train_y, epochs=5, callbacks=[mycallback])

# MAKING PREDICTION
predictions = model.predict(test_x)

# EVALUATING MODEL
print(model.evaluate(test_x, test_y))

# CASE 1
# When we use 1 convolution layer, 1 max pooling layer
# and Dense layer with 256 neurons the accuracy reached is
# 95.17 on training dataset and on testing dataset 91.38 (DISABLING THE CALLBACK)
# It took less time for training as compared to multiple convolution layer

# CASE 2
# When we use 2 Convolution layer, 2 Max Pooling layer
# and Dense layer with 128 neurons the accuracy achieved is
# 92.8 on training dataset and on testing dataset 90.95

# Considering both the cases we can say that always it is not beneficial to increase number
# of convolution layer sometimes by increasing number of neurons can also affect
# the performance and it is also computationally cheaper
