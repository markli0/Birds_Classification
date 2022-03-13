import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import pandas as pd
import os
from skimage import io
import cv2

DATASET_SIZE = 38562

def resize_images():
    folder_path = os.path.join(os.getcwd(), 'birds')
    dict = pd.read_csv(os.path.join(folder_path, 'labels.csv'))
    folder_path = os.path.join(folder_path, 'train')

    images = []
    labels = []
    counter = 0
    for (path, label) in zip(dict['path'], dict['class']):
        print(counter)
        counter += 1
        labels.append(label)
        img_path = os.path.join(folder_path, str(label))
        img_path = os.path.join(img_path, path)

        img = io.imread(img_path)  # solve format warnings
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        img = img[:, :, 0:3]  # remove additional layer


        img_tensor = tf.convert_to_tensor(img)
        img_tensor = tf.image.resize(img_tensor, (224, 224))  # Resizing the image to 224x224 dimension

        # images.append(tf.keras.preprocessing.image.img_to_array(img_tensor))
        new_img_path = os.path.join(os.getcwd(), 'train_ds')
        new_img_path = os.path.join(new_img_path, str(label))
        if not os.path.exists(new_img_path):
            os.makedirs(new_img_path)
        new_img_path = os.path.join(new_img_path, path[0:-3] + 'jpg')
        tf.keras.utils.save_img(new_img_path, img_tensor)


    # train_ds = tf.data.Dataset.from_tensor_slices((images, labels))
    # tf.data.experimental.save(train_ds, os.path.join(folder_path, 'resized_ds'))
    # new_dataset = tf.data.experimental.load(os.path.join(folder_path, 'resized_ds'))

    print(1)


def resize_images_test():
    folder_path = os.path.join(os.getcwd(), 'birds')
    folder_path = os.path.join(folder_path, 'test')
    folder_path = os.path.join(folder_path, '0')

    for path in os.listdir(folder_path):
        img_path = os.path.join(folder_path, path)

        img = io.imread(img_path)  # solve format warnings
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        img = img[:, :, 0:3]  # remove additional layer


        img_tensor = tf.convert_to_tensor(img)
        img_tensor = tf.image.resize(img_tensor, (224, 224))  # Resizing the image to 224x224 dimension

        # images.append(tf.keras.preprocessing.image.img_to_array(img_tensor))
        new_img_path = os.path.join(os.getcwd(), 'test_ds')
        new_img_path = os.path.join(new_img_path, '0')

        if not os.path.exists(new_img_path):
            os.makedirs(new_img_path)
        new_img_path = os.path.join(new_img_path, path)
        tf.keras.utils.save_img(new_img_path, img_tensor)


def load_datasets():
    folder_path = os.path.join(os.getcwd(), 'train_ds')

    builder = tfds.ImageFolder(folder_path)
    print(builder.info)  # num examples, labels... are automatically calculated
    train_ds = builder.as_dataset(split='train', shuffle_files=True, as_supervised=True)

    return train_ds.map(train_prep)


def load_test_ds():
    folder_path = os.path.join(os.getcwd(), 'test_ds')
    folder_path = os.path.join(folder_path, 'test')
    folder_path = os.path.join(folder_path, '0')
    paths = []
    imgs = []
    for path in os.listdir(folder_path):
        paths.append('test/' + path)
        img_path = os.path.join(folder_path, path)
        img = io.imread(img_path)
        img = np.array(img)
        img = img / 255
        img = img.reshape(1, 224, 224, 3)
        imgs.append(img)

    return paths, imgs


def load_maps():
    file = os.path.join(os.getcwd(), 'map.csv')
    list = pd.read_csv(file, index_col=False)
    list = np.array(list)
    maps = {}

    for i in range(len(list)):
        maps[list[i][1]] = list[i][0]

    return maps


def load_train_ds(c):
    folder_path = os.path.join(os.getcwd(), 'train_ds')
    folder_path = os.path.join(folder_path, 'train')
    folder_path = os.path.join(folder_path, str(c))
    paths = []
    imgs = []
    for path in os.listdir(folder_path):
        paths.append('test/' + path)
        img_path = os.path.join(folder_path, path)
        img = io.imread(img_path)
        img = np.array(img)
        img = img / 255
        img = img.reshape(1, 224, 224, 3)
        imgs.append(img)

    return paths, imgs


def train_prep(image, label):
    image = tf.cast(image, tf.float32) / 255
    image = tf.image.random_flip_left_right(image)
    image = tf.pad(image, [[4, 4], [4, 4], [0, 0]], 'REFLECT')
    image = tf.image.random_crop(image, (224, 224, 3))
    return image, label


def test_prep(x, y):
    x = tf.cast(x, tf.float32) / 255.
    return x, y
