import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf

from keras.applications import ResNet50 
from keras.optimizers import Adam   # type: ignore
from keras.models import Sequential # type: ignore
from keras import layers

import pandas as pd

import numpy

epochs = 5
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
        category = file.relative_to(data_path).parts[0] if file.relative_to(data_path).parts else None
        label = categories.index(category)
        
        if label is not None:
            files.append(str(file.resolve()))
            labels.append(label)
        
    return (categories, labels, files) 

def handle_image(file):
    image = tf.io.read_file(file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    image = image / 255.0
    return image

data_augmentation = Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

def train_ai():
    data_dir = Path(f'{os.getcwd()!s}/Data')
    
    categories, labels, files = load_data(data_dir)

    x_train, x_test, y_train, y_test = train_test_split(
        files, labels, test_size=0.2, random_state=42, stratify=labels
    )
        
    label_binarizer = LabelBinarizer()
    
    x_train = numpy.array(x_train)
    x_test = numpy.array(x_test)
    
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.transform(y_test)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(lambda x, y: (handle_image(x), y))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(lambda x, y: (handle_image(x), y))

    train_dataset = train_dataset.shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

    base_model.trainable = False

    model = Sequential([
        layers.Input(shape=(img_height, img_width, 3)),
        data_augmentation,
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(categories), activation='softmax')
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', 'precision'])

    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=epochs
    )
    
    print(model.summary())
    
    pd.DataFrame.from_dict(history.history).to_csv('history.csv', index=False)
    
    model.save("alzheimers_res.keras")