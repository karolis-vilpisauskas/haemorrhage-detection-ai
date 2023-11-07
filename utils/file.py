import os
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import numpy as np

ROOT_PATH = './data'
FILE_EXTENSIONS = ('jpg', 'JPG', 'png', 'PNG', 'tif', 'gif', 'ppm')
BATCH_SIZE = 2

DEFAULT_IMAGE_HEIGHT = 1280
DEFAULT_IMAGE_WIDTH = 1280
IMAGE_DIMENSION = 256
HEIGHT_IMAGE_COUNT = 5
HEIGHT_INCREMENT = 256
WIDTH_INCREMENT = 256
WIDTH_IMAGE_COUNT = 5


def getSortedFilePaths(path):
    pathToFiles = os.path.join(ROOT_PATH, path)
    return sorted(
        [
            os.path.join(pathToFiles, fname)
            for fname in os.listdir(pathToFiles)
            if fname.endswith(FILE_EXTENSIONS) and not fname.startswith(".")
        ]
    )


def transformMask(maskPath, data_format):
    mask = tf.io.read_file(maskPath)
    mask = tf.io.decode_png(mask)
    mask = mask[:, :, :3]
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.reduce_mean(mask, axis=-1, keepdims=True)
    mask.set_shape([None, None, 1])

    masks = []
    for i in range(0, HEIGHT_IMAGE_COUNT):
        for j in range(0, WIDTH_IMAGE_COUNT):
            currentMask = tf.image.crop_to_bounding_box(
                mask, HEIGHT_INCREMENT * i, WIDTH_INCREMENT * j, IMAGE_DIMENSION, IMAGE_DIMENSION)
            # currentMask = tf.cast(currentMask, tf.float32)
            if data_format == 'channels_first':
                currentMask = tf.transpose(currentMask, [2, 0, 1])
            masks.append(currentMask)

    return masks


def transformImage(imagePath, data_format):
    image = tf.io.read_file(imagePath)
    image = tf.io.decode_png(image)
    image = tf.squeeze(image)
    image.set_shape([None, None, 3])

    # Apply random rotation, flip, and zoom
    # image = tf.keras.preprocessing.image.random_rotation(
    #     image, 20, row_axis=0, col_axis=1, channel_axis=2)
    # image = tf.keras.preprocessing.image.random_flip_left_right(image)
    # image = tf.keras.preprocessing.image.random_zoom(image, (0.8, 1.2))

    image = tf.image.adjust_gamma(image, gamma=2)

    image = tf.cast(image, tf.float32) / 255.0

    images = []
    for i in range(0, HEIGHT_IMAGE_COUNT):
        for j in range(0, WIDTH_IMAGE_COUNT):
            currentImage = tf.image.crop_to_bounding_box(
                image, HEIGHT_INCREMENT * i, WIDTH_INCREMENT * j, IMAGE_DIMENSION, IMAGE_DIMENSION)
            # currentImage = tf.cast(currentImage, tf.float32)
            if data_format == 'channels_first':
                currentImage = tf.transpose(currentImage, [2, 0, 1])
            images.append(currentImage)

    return images


def loadData(trainingImagePaths, trainingMaskPaths, data_format='channels_last'):
    images = transformImage(trainingImagePaths, data_format)
    masks = transformMask(trainingMaskPaths, data_format)
    return images, masks


def flattenImages(images, masks):
    images = tf.unstack(images)
    masks = tf.unstack(masks)
    return tf.data.Dataset.from_tensor_slices((images, masks))


def filter_black(image, mask):
    mask_sum = tf.reduce_sum(mask)
    # Filter out images that have a low pixel intensity
    return mask_sum > 0 and tf.math.reduce_max(image) > 0.1


def datasetGenerator(trainingImagePaths, trainingMaskPaths, data_format='channels_last'):
    dataset = tf.data.Dataset.from_tensor_slices(
        (trainingImagePaths, trainingMaskPaths))

    # Load and preprocess the images and masks
    dataset = dataset.map(lambda image_path, mask_path: loadData(
        image_path, mask_path, data_format))

    # Flatten the image and mask lists
    dataset = dataset.flat_map(flattenImages)

    # Filter out the black masks
    dataset = dataset.filter(filter_black)

    # Shuffle the dataset
    dataset = dataset.shuffle(buffer_size=200)

    # Batch the dataset
    dataset = dataset.batch(BATCH_SIZE)

    # Prefetch the dataset to improve performance
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def visualize(**images):
    n = len(images)
    plt.figure(figsize=(20, 20))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(1, n, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(' '.join(name.split('_')).title())
        plt.imshow(image, cmap='gray')
    plt.show()


def testDataset(train_dataset):
    image, mask = next(iter(train_dataset.take(1)))
    print(image.shape, mask.shape)

    for (img, msk) in zip(image, mask):
        print(mask.numpy().min(), mask.numpy().max())
        print(np.unique(mask.numpy()))
        visualize(
            nuotrauka=img.numpy(),
            anotacija=msk.numpy(),
        )
