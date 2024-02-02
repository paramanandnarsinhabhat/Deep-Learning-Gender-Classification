#!/usr/bin/env python
# coding: utf-8

import zipfile
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt

# Load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('/Users/paramanandbhat/Downloads/test_fkwGUNG.csv')
base_path = '/Users/paramanandbhat/Downloads/train_nLPp5K8/images'

# Update the image paths in the DataFrame
train_df['image_path'] = train_df['image_names'].apply(lambda x: os.path.join(base_path, x))
test_df['image_path'] = test_df['image_names'].apply(lambda x: os.path.join(base_path, x))

# Convert the 'class' column to string type for compatibility
train_df['class'] = train_df['class'].astype(str)

# Data augmentation configuration
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    vertical_flip=True,
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

# Define and compile the model with MobileNetV2 as the base model
base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Configure callbacks including ReduceLROnPlateau
callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='min'),
    ModelCheckpoint('best_model_simplified.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)
]

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,
    verbose=1,
    callbacks=callbacks
)

# Plot training and validation accuracy and loss
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

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
preprocessed_images = np.vstack([tf.keras.applications.mobilenet_v2.preprocess_input(img_to_array(load_img(path, target_size=(224, 224))).reshape(1, 224, 224, 3)) for path in test_df['image_path']])

# Make predictions
predictions = model.predict(preprocessed_images)
predicted_classes = (predictions > 0.5).astype(int).reshape(-1)

# Prepare submission
test_df['class'] = predicted_classes
submission_df = test_df[['image_names', 'class']]
submission_df.to_csv('/Users/paramanandbhat/Downloads/train_nLPp5K8/submissionfinal5.csv', index=False)
