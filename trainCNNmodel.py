import os
import cv2
import numpy as np
import random
from matplotlib import pyplot as plt
from tensorflow import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Input, Dropout, MaxPooling2D, Conv2D, Concatenate, Embedding, Reshape, Flatten, Activation, BatchNormalization
from keras.optimizers import SGD
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator

SOURCE_IMG_HIGHT = 480
SOURCE_IMG_WIDTH = 640
HEIGHT = 240
WIDTH = 320
HEIGHT_REQUIRED_PORTION = 0.5
WIDTH_REQUIRED_PORTION = 0.9
MAX_STEER_DEGREES = 40
data_dir = 'C:/SelfDrive/GPS with Vision/_img'

height_from = int(HEIGHT * (1 - HEIGHT_REQUIRED_PORTION))
width_from = int((WIDTH - WIDTH * WIDTH_REQUIRED_PORTION) / 2)
width_to = width_from + int(WIDTH_REQUIRED_PORTION * WIDTH)

new_height = HEIGHT - height_from
new_width = width_to - width_from

batch_size = 64
image_size = (new_width, new_height)

label_position = -5

def custom_data_generator(image_files, batch_size):
    num_samples = len(image_files)
    while True:
        indices = np.random.randint(0, num_samples, batch_size)
        batch_images = []
        batch_input_2 = []
        batch_labels = []
        for idx in indices:
            image_path = image_files[idx]
            label = float(os.path.basename(image_path).split('.png')[0].split('_')[2])
            if label > MAX_STEER_DEGREES:
                label = MAX_STEER_DEGREES
            elif label < -MAX_STEER_DEGREES:
                label = -MAX_STEER_DEGREES
            label = float(label) / MAX_STEER_DEGREES
            input_2 = int(os.path.basename(image_path).split('.png')[0].split('_')[1])
            image = preprocess_image(image_path)
            batch_images.append(image)
            batch_input_2.append(input_2)
            batch_labels.append(label)
        yield [np.array(batch_images), np.array(batch_input_2)], np.array(batch_labels)

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (WIDTH, HEIGHT))
    image = image[height_from:, width_from:width_to]
    image = image / 255.0
    return image

def create_model():
    image_input = Input(shape=(new_height, new_width, 3))
    integer_input = Input(shape=(1,))
    x = Conv2D(64, kernel_size=(6, 6), activation='relu', padding='same')(image_input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(6, 6), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(6, 6), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(64, kernel_size=(6, 6), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dense(8, activation='relu', activity_regularizer=regularizers.L2(1e-5))(x)
    x = Dropout(0.2)(x)
    x = Dense(4, activation='relu', activity_regularizer=regularizers.L2(1e-5))(x)
    x = Flatten()(x)
    concatenated_inputs = Concatenate()([x, integer_input])
    output = Dense(1, activation='linear')(concatenated_inputs)
    model = Model(inputs=[image_input, integer_input], outputs=output)
    return model

image_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir) if file.endswith('.png')]

random.shuffle(image_files)

split_index = int(len(image_files) * 0.8)
train_files, val_files = image_files[:split_index], image_files[split_index:]

train_generator = custom_data_generator(train_files, batch_size)
val_generator = custom_data_generator(val_files, batch_size)

model = create_model()
model.summary()
model.compile(loss='MSE', optimizer='adam')

model.fit(train_generator, steps_per_epoch=len(train_files) // batch_size, epochs=10,
          validation_data=val_generator, validation_steps=len(val_files) // batch_size)

desired_layer_output = model.get_layer('dense').output
model_to_save = Model(inputs=model.input, outputs=desired_layer_output)

model_to_save.save('CNN_image_model.h5')
