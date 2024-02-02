#!/usr/bin/env python
# coding: utf-8

import zipfile
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

# Load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('/Users/paramanandbhat/Downloads/test_fkwGUNG.csv')
base_path = '/Users/paramanandbhat/Downloads/train_nLPp5K8/images'

# Update the image paths in the DataFrame
train_df['image_path'] = base_path + '/' + train_df['image_names'].astype(str)
test_df['image_path'] = base_path + '/' + test_df['image_names'].astype(str)

# Verify that an image file exists (optional)
print(os.path.exists(train_df['image_path'].iloc[0]))

# Convert the 'class' column to string type
train_df['class'] = train_df['class'].astype(str)

# Data augmentation
# Example using ImageDataGenerator with additional augmentations
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,  # New augmentation
    fill_mode='nearest',
    validation_split=0.2
)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

# Prepare the data generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_path',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

validation_generator = val_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_path',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Define the model
# Simplified Model Architecture
# Adding more convolutional layers and increasing depth
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),  # New layer
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),  # Increased units
    Dropout(0.5),  # Keep high dropout to combat overfitting
    Dense(1, activation='sigmoid')
])

# Compile the model
# Compile the model with a potentially lower initial learning rate
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),  # Adjusted learning rate
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Extended patience for EarlyStopping
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min'),  # Increased patience
    ModelCheckpoint('best_model_simplified.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
]

# Using ReduceLROnPlateau for dynamic learning rate adjustments
from tensorflow.keras.callbacks import ReduceLROnPlateau

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)

callbacks.append(reduce_lr)


# Increased training epochs with EarlyStopping in place
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,  # Allowing for more epochs with early stopping
    verbose=1,
    callbacks=callbacks
)


# Plot training and validation accuracy and loss
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

plot_training_history(history)

# Preprocess and predict on test images
def preprocess_image(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet_v2.preprocess_input(img_array_expanded_dims)

preprocessed_images = np.vstack([preprocess_image(path) for path in test_df['image_path']])

# Make predictions
predictions = model.predict(preprocessed_images)
predicted_classes = (predictions > 0.5).astype(int).reshape(-1)

# Prepare submission
test_df['class'] = predicted_classes
submission_df = test_df[['image_names', 'class']]
submission_df.to_csv('/Users/paramanandbhat/Downloads/train_nLPp5K8/submissionfinal4.csv', index=False)
