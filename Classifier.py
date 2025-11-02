import tensorflow as tf
from tensorflow.keras import layers, models 
from tensorflow.keras.applications import MobileNetV2 #Unneeded for now
import matplotlib.pyplot as plt
import numpy as np

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

train_labels = train_labels.flatten()
test_labels = test_labels.flatten()

animate_labels = [2, 3, 4, 5, 6, 7]

train_labels = np.isin(train_labels, animate_labels).astype(int)
test_labels = np.isin(test_labels, animate_labels).astype(int)

train_images = train_images / 255.0
test_images = test_images / 255.0

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

train_ds = train_ds.shuffle(10000).batch(32)
test_ds = test_ds.batch(32)
dataset = (train_ds, test_ds)

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(train_ds, validation_data=test_ds, epochs=10)



