import os
from pathlib import Path

import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

def handle_image(file):
    image = tf.io.read_file(file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [256, 256])
    image = image / 255.0
    return image

# Testing on a Single Image
test_image_path = str(Path('/home/evie/project-ai/Data/Moderate Dementia/OAS1_0308_MR1_mpr-1_105.jpg'))
test_image = handle_image(test_image_path)
test_image = tf.expand_dims(test_image, axis=0)  # Adding batch dimension

def predict_ai():
    model = load_model("alzheimers_res.keras")
    
    # this is the order my path reading func reads the categories, if the order is changed, results would be skewed
    categories = ['Very mild Dementia', 'Mild Dementia', 'Moderate Dementia', 'Non Demented']

    # Make a prediction
    prediction = model.predict(test_image)
    predicted_class_index = np.argmax(prediction, axis=1)[0]  # Get the predicted class index
    pred_label = categories[predicted_class_index]  # Map the index to the corresponding label

    print(f"Predicted Class Index: {predicted_class_index}")
    print(f"Predicted Label: {pred_label}")

    # Display Image + Prediction
    plt.imshow(plt.imread(test_image_path))
    plt.title(f"Predicted Label: {pred_label}")
    plt.axis('off')
    plt.show()

