import zipfile
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, legacy
import matplotlib.pyplot as plt
import numpy as np

# Unzip files
with zipfile.ZipFile('/Users/paramanandbhat/Downloads/train_nLPp5K8.zip', 'r') as zip_ref:
    zip_ref.extractall(".")

# Load datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('/Users/paramanandbhat/Downloads/test_fkwGUNG.csv')
base_path = '/Users/paramanandbhat/Downloads/train_nLPp5K8/images'

# Update image paths
train_df['image_path'] = train_df['image_names'].apply(lambda x: f'{base_path}/{x}.jpg')
train_df['class'] = train_df['class'].astype(str)

# Prepare data generators
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=15, width_shift_range=0.1,
                                   height_shift_range=0.1, shear_range=0.1, zoom_range=0.1,
                                   horizontal_flip=True, fill_mode='nearest', validation_split=0.2)

val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = train_datagen.flow_from_dataframe(dataframe=train_df, x_col='image_path',
                                                    y_col='class', target_size=(224, 224),
                                                    batch_size=32, class_mode='binary', subset='training')

validation_generator = val_datagen.flow_from_dataframe(dataframe=train_df, x_col='image_path',
                                                       y_col='class', target_size=(224, 224),
                                                       batch_size=32, class_mode='binary', subset='validation')

# Model definition
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

model.compile(optimizer=legacy.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Model training
callbacks = [EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min'),
             ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1)]

history = model.fit(train_generator, validation_data=validation_generator, epochs=20, verbose=1, callbacks=callbacks)

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
test_df['image_path'] = test_df['image_names'].apply(lambda x: f'{base_path}/{x}.jpg')
preprocessed_images = np.vstack([tf.keras.applications.mobilenet_v2.preprocess_input(img_to_array(load_img(path, target_size=(224, 224))).reshape(1, 224, 224, 3)) for path in test_df['image_path']])

predictions = model.predict(preprocessed_images)
test_df['class'] = (predictions > 0.5).astype(int).reshape(-1)

# Save submission
submission_df = test_df[['image_names', 'class']]
submission_df.to_csv('/Users/paramanandbhat/Downloads/train_nLPp5K8/submissionfinal.csv', index=False)
