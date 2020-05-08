import matplotlib.pyplot as plt
import tensorflow as tf
from mlxtend.data import loadlocal_mnist

train_x, train_y = loadlocal_mnist(images_path='train-images-idx3-ubyte'
                                , labels_path='train-labels-idx1-ubyte')
test_x, test_y = loadlocal_mnist(images_path='t10k-images-idx3-ubyte',
                                 labels_path='t10k-labels-idx1-ubyte')
train_x = train_x.reshape(60000,28,28)
test_x = test_x.reshape(10000,28,28)
train_x, test_x = train_x/255.0 , test_x/255.0

plt.imshow(train_x[25060])
plt.show()

modnn = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

modnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

modnn.fit(train_x, train_y, epochs=5)

score = modnn.evaluate(test_x, test_y)
print("The loss is",score[0])
print("The accuracty is", score[1])
mylabel = ['tshirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Something', "Don't know it's name"]
# Enter a number <=10000 in this function and it will give you the label, It takes model as an input
n = int(input("Enter any number to test "))
plt.imshow(test_x[n])
plt.show()
pred = modnn.predict(test_x[n].reshape(1,28,28))
print("According to Neural Network the image is of",mylabel[pred.argmax()])