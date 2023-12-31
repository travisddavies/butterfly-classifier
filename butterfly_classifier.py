import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial
import numpy as np

# Image dimensions and batch size for training.
img_width = 224
img_height = 224
batch_size = 32

# Image augmentation for train data.
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.3,
    height_shift_range=0.2,
    zoom_range=[1.3, 0.7],
    shear_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
    )

# Generator for test and validation data.
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255)

# Getting the train data.
train_data = train_datagen.flow_from_directory(
        directory='train',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

# Getting the validation data.
val_data = test_datagen.flow_from_directory(
        directory='val',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

# Getting the test data.
test_data = test_datagen.flow_from_directory(
        directory='test',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

# Getting the number of classes in the data set.
num_classes = len(train_data.class_indices.keys())

tf.random.set_seed(42)  # extra code – ensures reproducibility

# Default convolutional layer for the neural net.
DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, padding="same",
                        activation="relu", kernel_initializer="he_normal")

# The architecture of the convolutional neural net.
model = tf.keras.Sequential([
    DefaultConv2D(filters=32, kernel_size=7,
                  input_shape=[img_width, img_height, 3]),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=64),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=128),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=256),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=256, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=num_classes, activation="softmax")
])

# The optimiser of the neural net - Adam with eta of 4e-4
optimiser = tf.keras.optimizers.Adam(learning_rate=2e-4)

# Compiling the neural net.
model.compile(loss="categorical_crossentropy", optimizer=optimiser,
              metrics=["accuracy"])

# early stopping with a patience of 30 epochs, which returns the best weights
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=50, restore_best_weights=True)

# train the model with 200 epochs
history_train = model.fit(train_data, validation_data=val_data,
                          epochs=300, callbacks=[early_stopping_cb])

# Plot the history of the epochs for the neural net.
pd.DataFrame(history_train.history).plot(
    xlim=[0, 300], ylim=[0, 1.2],
    grid=True, xlabel='Epoch', style=['r--', 'r--.', 'p-', 'b-*'])

# Turn validation data intro training data with augmentation
val_data = train_datagen.flow_from_directory(
        directory='val',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

# Train on the validation data
history_val = model.fit(val_data, epochs=30)

# Predict the test data
probabilities = model.predict(test_data)

# Saving the model
save_dir = 'saved_model'
model.save(save_dir)

print('Generating submission.csv file...')

test_yhat = np.argmax(probabilities, axis=1)

# Write the submission file
np.savetxt(
    'submission.csv',
    test_yhat,
    fmt='%d',
    delimiter=',',
    header='label',
    comments='')

# Show the plot.
plt.show()
