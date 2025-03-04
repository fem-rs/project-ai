# references: https://www.tensorflow.org/tutorials/images/classification

import os
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import tensorflow as tf

from keras.optimizers import Adam   # type: ignore
from keras.models import Sequential # type: ignore
from keras import layers

import numpy

epochs = 10
batch_size = 32
img_height = 300
img_width = 300


def load_data(data_directory):
    categories, labels, files = [], [], []
    
    for child in data_directory.iterdir():
        categories.append(str(child).removeprefix(f'{os.getcwd()}\\Data\\'))
        
    for file in Path(f"{os.getcwd()}\\Data").glob("**/*"):
        files.append(str(file))
        labels.append(categories.index(str(file).removeprefix(f'{os.getcwd()!s}\\Data\\').split('\\')[0]))
        
    return (categories, labels, files) 

def handle_image(file):
    image = tf.io.read_file(file)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    image = image / 255.0
    return image
     

def train_ai():
    data_dir = Path(f'{os.getcwd()!s}\\Data')
    
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
    
    model = Sequential([
        layers.Input(shape=(img_height, img_width, 3)),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(len(categories))
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

    model.save("alzheimers.keras")