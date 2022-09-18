import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
import secrets

print("TensorFlow version: ", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

data_dir = "yoga32"
train_batch_size = 32
rand_seed = secrets.randbelow(1_000_000_000)  # random seed for train/val split
# note that same seed must be used for both to ensure no overlap in train/val data

# Get training images from 'train' directory
train_data = tf.keras.utils.image_dataset_from_directory(
    data_dir + '/train',
    validation_split=0.2,
    subset="training",
    seed=rand_seed,
    image_size=(32, 32),
    batch_size=train_batch_size)

# Get validation images from 'train' directory
val_data = tf.keras.utils.image_dataset_from_directory(
    data_dir + '/train',
    validation_split=0.2,
    subset="validation",
    seed=rand_seed,
    image_size=(32, 32),
    batch_size=train_batch_size)

# Output list of class names
class_names = train_data.class_names
print(class_names)

for images, labels in val_data.take(1):
    print(images)
    print(labels)

# Plot training and validation accuracy of a model
def plot_model_history(history):
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.title('Training and validating accuracy')
    plt.legend()
    plt.show()


# 1. CNN Implementation
# 1.1 Basic architecture
cnn_model1 = keras.Sequential(
    [
        layers.Input((32, 32, 3)),

        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.Conv2D(16, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),

        layers.Conv2D(24, (3, 3), activation='relu'),
        layers.Conv2D(24, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2), strides=(2, 2)),

        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ],
)


# cnn_model1.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
# history1 = cnn_model1.fit(train_data, validation_data=val_data, epochs=10)
# plot_model_history(history1)

# Get test images from 'test' directory
test_data = tf.keras.utils.image_dataset_from_directory(
    data_dir+'/test',
    image_size=(32, 32),
    batch_size=7,
    shuffle=False,
)
