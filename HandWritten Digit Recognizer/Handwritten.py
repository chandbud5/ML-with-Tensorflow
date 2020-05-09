import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Loading Dataset
df = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")

train_y = df[['label']].values
X = df.values
X = X[:,1:]
train_x = X.reshape(42000, 28, 28, 1)

test_x = df_test.values.reshape(28000,28,28,1)
plt.imshow(test_x[99].reshape(28,28))
plt.show()

# Preprocessing Dataset
train_x, test_x = train_x/255.0, test_x/255.0

# Creating Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64,(3,3),activation=tf.nn.relu, input_shape=(28,28,1)),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_x, train_y, epochs=2)

prediction = model.predict(test_x)
label = np.argmax(prediction,axis=1)
print(label.shape,label)
np.savetxt("submission.csv",label)

# Case 1
# Simple Neural Network with 2 layers 1st 512 neuron and 2nd 10 neurons with activation as softmax 5 - epochs
# Training accuracy is around 99%

# Case 2
# Convolution NN with 4 layers 1st Conv2D(64,(3,3)), 2nd MaxPooling2D(2,2), 3rd Dense with 128, 4th Dense 10 - 3 epochs
# Training Accuracy - 98%