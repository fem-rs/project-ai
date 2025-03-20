import os
from pathlib import Path

import tensorflow as tf
from keras.applications import ResNet50
from keras.optimizers import Adam
from keras.models import Sequential
from keras import layers

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE


epochs = 50
batch_size = 64
img_height = 256
img_width = 256

def load_data(data_directory):
    categories, labels, files = [], [], []
    data_path = Path(data_directory).resolve(strict=True)
    
    for child in data_path.iterdir():
        if child.is_dir():
            categories.append(child.name)

    for file in data_path.rglob("*.*"):
        category = file.relative_to(data_path).parts[0]
        if category in categories:
            label = categories.index(category)
            files.append(str(file.resolve()))
            labels.append(label)

    return categories, labels, files

def handle_image(file, label):
    image = tf.io.read_file(file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    image = image / 255.0
    return image, label

data_augmentation = Sequential([
    layers.RandomFlip(),
    layers.RandomRotation(0.2),
])


def train_ai():
    data_dir = Path(f'{os.getcwd()}/Data') # Not cross-platform friendly, should probably change

    categories, labels, files = load_data(data_dir)
    label_binarizer = LabelBinarizer()

    x_train, x_test, y_train, y_test = train_test_split(
        files, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.transform(y_test)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(labels),
        y=labels
    )
    
    class_weights = {i: class_weights[i] for i in range(len(class_weights))} 
         
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(handle_image)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(handle_image)

    # Apparently having the shuffle the same as the amount of files in the dataset is best practice 
    # but my laptop doesn't have a strong enough gpu
    train_dataset = train_dataset.shuffle(3000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    base_model.trainable = True 

    for layer in base_model.layers[:10]: # make untrainable for first 10 images, what the fuck is this indexing syntax i hate python
        layer.trainable = False    

    model = Sequential([
        layers.Input(shape=(img_height, img_width, 3)),
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.6),
        layers.Dense(len(categories), activation='softmax')
    ])

    # TODO: i want to add a learning rate scheduler to fine tune stuff more
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=tf.keras.losses.CategoricalFocalCrossentropy(),
                  metrics=['accuracy', 'precision', 'recall', 'f1_score', 'false_positives', 'false_negatives'])

    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs,
        class_weight=class_weights, # hopefully now it wont cum when it sees Very mild dementia
    )

    print(model.summary())

    pd.DataFrame.from_dict(history.history).to_csv('history.csv', index=False) # needed for graphing accuracy, etc in frontend
    model.save("alzheimers.keras")
