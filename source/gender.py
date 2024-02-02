# Unzip the files
import zipfile

with zipfile.ZipFile('/Users/paramanandbhat/Downloads/train_nLPp5K8.zip', 'r') as zip_ref:
    zip_ref.extractall(".")

import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import legacy
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np

# Load the data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('/Users/paramanandbhat/Downloads/test_fkwGUNG.csv')
base_path = '/Users/paramanandbhat/Downloads/train_nLPp5K8/images'

# Create a train-validation split
train_df['image_path'] = train_df['image_names'].apply(lambda x: f'{base_path}/{x}.jpg')
train, val = train_test_split(train_df, test_size=0.2)

# Image data generators
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# Convert the 'class' column to string type
train_df['class'] = train_df['class'].astype(str)

# Create the train_generator and validation_generator
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train,
    x_col='image_path',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

validation_generator = val_datagen.flow_from_dataframe(
    dataframe=val,
    x_col='image_path',
    y_col='class',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=legacy.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=20,
    verbose=1
)

# Plot training and validation accuracy and loss
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

# Prepare test images
test_df['image_path'] = test_images_directory + test_df['image_names'].astype(str)
preprocessed_images = np.vstack([preprocess_image(path) for path in test_df['image_path']])

# Make predictions
predictions = model.predict(preprocessed_images)
predicted_classes = (predictions > 0.5).astype(int).reshape(-1)

# Prepare submission
test_df['class'] = predicted_classes
submission_df = test_df[['image_names', 'class']]
submission_df.to_csv('/Users/paramanandbhat/Downloads/train_nLPp5K8/submissionfinal.csv', index=False)

