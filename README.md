# traffic-light-recognition
# Traffic Light Recognition Problem Definition:
# The goal is to classify images of traffic lights into different categories, such as:

# Red: Indicates stop.
# Yellow: Indicates to slow down or prepare to stop.
# Green: Indicates go.
# Off: The traffic light is off or not functioning.
# For this problem, it is needed to use a dataset of traffic light images and build a CNN model to predict the state of the traffic light in an image.

# Steps:
# Problem Definition: Understanding that we need to classify traffic light images into categories (Red, Yellow, Green, Off).
# Dataset Preparation: Download a dataset of traffic light images.
# Data Preprocessing: Resize images, normalize, and split into training and test datasets.
# Model Training: Build a CNN model to classify the images.
# Model Evaluation: Evaluate the model's accuracy on the test set.
# Prediction: Use the model to make predictions on new images.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2  # OpenCV for image processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 2: Load the dataset
# it is needed to have a traffic light dataset with images of traffic lights labeled as 'Red', 'Yellow', 'Green', or 'Off'.
# For this example, assuming that the images are stored in directories: 'red/', 'yellow/', 'green/', 'off/'
# Dataset folder structure:
# - traffic_light_data/
#     - red/
#     - yellow/
#     - green/
#     - off/

# Define the paths
dataset_dir = 'traffic_light_data'  # Change to the actual path of your dataset

# Step 3: Load and preprocess the images

image_size = (64, 64)  # Resize images to 64x64 for CNN input
categories = ['red', 'yellow', 'green', 'off']

data = []
labels = []

for category in categories:
    category_path = os.path.join(dataset_dir, category)
    
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        
        # Load image, resize, and normalize
        img = cv2.imread(img_path)
        img = cv2.resize(img, image_size)  # Resize image to 64x64
        img = img / 255.0  # Normalize image pixels to range [0, 1]
        
        data.append(img)
        labels.append(category)

# Convert lists to numpy arrays
X = np.array(data)
y = np.array(labels)

# Step 4: Encode the labels
# Use LabelEncoder to convert string labels into integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# One-hot encode the labels
y = to_categorical(y, num_classes=len(categories))

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

# Step 6: Build the CNN model
model = Sequential()

# Convolutional layers to extract features from images
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the feature maps and connect to a dense layer
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout to avoid overfitting

# Output layer with softmax activation for multi-class classification
model.add(Dense(len(categories), activation='softmax'))

# Step 7: Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 8: Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 9: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Model Accuracy on Test Data: {accuracy * 100:.2f}%')

# Step 10: Visualize the training process (loss and accuracy)
# Plot the loss curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Test Loss')
plt.title('Loss Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot the accuracy curve
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Test Accuracy')
plt.title('Accuracy Curve')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Step 11: Use the model to make predictions on new images
def predict_traffic_light(img_path):
    # Load the image and preprocess it
    img = cv2.imread(img_path)
    img = cv2.resize(img, image_size)  # Resize to (64, 64)
    img = img / 255.0  # Normalize image pixels to range [0, 1]
    
    # Add an extra dimension for batch size (model expects batch of images)
    img = np.expand_dims(img, axis=0)
    
    # Predict the class
    prediction = model.predict(img)
    
    # Get the class label
    predicted_class = label_encoder.inverse_transform([np.argmax(prediction)])[0]
    
    return predicted_class

# Example usage: Predict the traffic light state from an image
img_path = 'path_to_new_traffic_light_image.jpg'  # Replace with actual image path
predicted_state = predict_traffic_light(img_path)
print(f"Predicted Traffic Light State: {predicted_state}")
Explanation of the Code:
Dataset Preparation:

The traffic light images are assumed to be stored in separate folders (red, yellow, green, off) within a main directory. Each folder contains images labeled according to the traffic light state.
Image Preprocessing:

Each image is resized to 64x64 pixels and normalized to a range between 0 and 1.
We also encode the labels (i.e., the traffic light states) using LabelEncoder and one-hot encode the labels for multi-class classification.
Model Architecture:

A simple CNN model is used to classify the images. The CNN consists of:
Three convolutional layers with ReLU activation followed by max-pooling layers.
A Flatten layer to convert the 2D feature maps into a 1D vector.
A Dense layer with 128 neurons and ReLU activation.
A final Dense layer with softmax activation for multi-class classification.
Model Training:

The model is trained using the Adam optimizer and categorical cross-entropy loss function. It is evaluated on the test dataset to calculate the accuracy.
Model Evaluation:

The accuracy and loss are plotted for both the training and validation sets to monitor the training progress and evaluate performance.
Prediction:

The predict_traffic_light() function can be used to predict the state of a traffic light from a new image by preprocessing it and passing it through the trained CNN model.
