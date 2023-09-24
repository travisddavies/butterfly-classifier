import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from functools import partial

img_width = 128
img_height = 128
batch_size = 32

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.3,
    height_shift_range=0.2,
    zoom_range=[1.3, 0.7],
    shear_range=0.1,
    # channel_shift_range=0.4,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
    )

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255)

train_data = train_datagen.flow_from_directory(
        directory='train',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

val_data = test_datagen.flow_from_directory(
        directory='val',
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

num_classes = len(train_data.class_indices.keys())

tf.random.set_seed(42)  # extra code – ensures reproducibility
DefaultConv2D = partial(tf.keras.layers.Conv2D, kernel_size=3, padding="same",
                        activation="relu", kernel_initializer="he_normal")
model = tf.keras.Sequential([
    DefaultConv2D(filters=32, kernel_size=7,
                  input_shape=[img_width, img_height, 3]),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=64),
    tf.keras.layers.MaxPool2D(),
    DefaultConv2D(filters=128),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=128, activation="relu",
                          kernel_initializer="he_normal"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=num_classes, activation="softmax")
])
optimiser = tf.keras.optimizers.Adam(learning_rate=4e-4)
model.compile(loss="categorical_crossentropy", optimizer=optimiser,
              metrics=["accuracy"])

# early stopping with a patience of 30 epochs, which returns the best weights
early_stopping_cb = tf.keras.callbacks.EarlyStopping(
    patience=50, restore_best_weights=True)

# train the model with 200 epochs
history = model.fit(train_data, validation_data=val_data, epochs=400,
                    callbacks=[early_stopping_cb])

pd.DataFrame(history.history).plot(
    xlim=[0, 300], ylim=[0, 1.2],
    grid=True, xlabel='Epoch', style=['r--', 'r--.', 'p-', 'b-*'])

plt.show()
