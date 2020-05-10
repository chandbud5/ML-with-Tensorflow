import os
import zipfile
import tensorflow as tf
import matplotlib.pyplot as plt

# ref = zipfile.ZipFile("cats_and_dogs_filtered.zip")
# ref.extractall("\pro\ML_with_tensorflow\Cat and Dog\dataset")

# print(os.listdir("\pro\ML_with_tensorflow\Cat and Dog\dataset\cats_and_dogs_filtered"))

train_dir =  # Path to your training images directory
validation_dir = # Path to your validation images directory


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


model.summary()

model.compile(optimizer=tf.optimizers.RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=(1/255.0),
                                                            width_shift_range=0.2,
                                                            height_shift_range=0.2,
                                                            shear_range=0.2,
                                                            rotation_range=40,
                                                            zoom_range=0.2,
                                                            horizontal_flip=True,
                                                            fill_mode="nearest")

validation_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=(1/255.0),
                                                                 width_shift_range=0.2,
                                                                 height_shift_range=0.2,
                                                                 shear_range=0.2,
                                                                 zoom_range=0.2,
                                                                 rotation_range=40,
                                                                 horizontal_flip=True,
                                                                 fill_mode="nearest")


train_generator = train_gen.flow_from_directory(train_dir,
                                                target_size=(150,150),
                                                batch_size=20,
                                                class_mode="binary")

validation_generator = validation_gen.flow_from_directory(validation_dir,
                                                          target_size=(150,150),
                                                          batch_size=20,
                                                          class_mode="binary")

history = model.fit(train_generator,
          steps_per_epoch=100,
          epochs=20,
          validation_data=validation_generator,
          validation_steps=50,
          verbose=1)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epo = range(len(acc))

plt.plot(epo, acc,label="Train")
plt.plot(epo, val_acc, label="Validation")
plt.title("Accuracy v/s epochs")
plt.legend()
plt.show()

plt.plot(epo, loss, label="train")
plt.plot(epo, val_loss, label="validation")
plt.title("Loss v/s Epochs")
plt.legend()
plt.show()
