import os
import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
import numpy as np

ROOT_PATH = './data'
FILE_EXTENSIONS = ('jpg', 'JPG', 'png', 'PNG', 'tif', 'gif', 'ppm')
BATCH_SIZE = 2

DEFAULT_IMAGE_HEIGHT = 2848
DEFAULT_IMAGE_WIDTH = 4288
IMAGE_DIMENSION = 512
HEIGHT_IMAGE_COUNT = 9
HEIGHT_INCREMENT = 292
WIDTH_INCREMENT = 472
WIDTH_IMAGE_COUNT = 9


def getSortedFilePaths(path):
    pathToFiles = os.path.join(ROOT_PATH, path)
    return sorted(
        [
            os.path.join(pathToFiles, fname)
            for fname in os.listdir(pathToFiles)
            if fname.endswith(FILE_EXTENSIONS) and not fname.startswith(".")
        ]
    )


def transformMask(maskPath):
    mask = tf.io.read_file(maskPath)
    mask = tfio.experimental.image.decode_tiff(mask)
    mask = mask[:, :, :3]
    mask = tf.image.rgb_to_grayscale(mask)
    mask = tf.divide(mask, 76)
    mask.set_shape([None, None, 1])

    masks = []
    for i in range(0, HEIGHT_IMAGE_COUNT):
        for j in range(0, WIDTH_IMAGE_COUNT):
            currentMask = tf.image.crop_to_bounding_box(
                mask, HEIGHT_INCREMENT * i, WIDTH_INCREMENT * j, IMAGE_DIMENSION, IMAGE_DIMENSION)
            currentMask = tf.cast(currentMask, tf.int32)
            masks.append(currentMask)

    return masks


def transformImage(imagePath):
    image = tf.io.read_file(imagePath)
    image = tf.io.decode_jpeg(image)
    image = tf.squeeze(image)
    image.set_shape([None, None, 3])
    image = tf.image.adjust_contrast(image, 2.5)

    images = []
    for i in range(0, HEIGHT_IMAGE_COUNT):
        for j in range(0, WIDTH_IMAGE_COUNT):
            currentImage = tf.image.crop_to_bounding_box(
                image, HEIGHT_INCREMENT * i, WIDTH_INCREMENT * j, IMAGE_DIMENSION, IMAGE_DIMENSION)
            currentImage = tf.cast(currentImage, tf.int32)
            images.append(currentImage)

    return images


def loadData(trainingImagePaths, trainingMaskPaths):
    images = transformImage(trainingImagePaths)
    masks = transformMask(trainingMaskPaths)
    return images, masks


def flattenImages(images, masks):
    images = tf.unstack(images)
    masks = tf.unstack(masks)
    return tf.data.Dataset.from_tensor_slices((images, masks))


def datasetGenerator(trainingImagePaths, trainingMaskPaths):
    dataset = tf.data.Dataset.from_tensor_slices(
        (trainingImagePaths, trainingMaskPaths))
    dataset = dataset.map(loadData)
    dataset = dataset.flat_map(flattenImages)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=False)
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
