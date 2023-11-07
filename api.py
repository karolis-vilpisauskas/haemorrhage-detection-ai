from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from utils import file

app = FastAPI()

DEFAULT_IMAGE_HEIGHT = 2848
DEFAULT_IMAGE_WIDTH = 4288
IMAGE_DIMENSION = 512
HEIGHT_IMAGE_COUNT = 9
HEIGHT_INCREMENT = 292
WIDTH_INCREMENT = 472
WIDTH_IMAGE_COUNT = 9

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Image(BaseModel):
    src: str

    class Config:
        orm_mode = True


@app.post("/transform-images")
async def transform_images(body: Image):
    encoded_data = body.src.split(',')[1]
    nparr = np.frombuffer(base64.urlsafe_b64decode(encoded_data), np.float32)
    decodedjpeg = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    cv2.imwrite("./tmp/file.jpeg", decodedjpeg)
    dataset = file.datasetGenerator(
        ["./tmp\\file.jpeg"], ["./data\\training/masks\\IDRiD_01_HE.tif"])
    model = tf.keras.models.load_model('./unet-model')
    prediction = None
    for image, mask in dataset.take(1):
        prediction = model.predict(image)[0]
        print(prediction)
        # prediction = base64.b64encode(prediction)

    # image = tf.io.read_file("./tmp/file.jpeg")
    # image = tf.io.decode_jpeg(image)
    # image = tf.squeeze(image)
    # image.set_shape([None, None, 3])
    # image = tf.image.adjust_contrast(image, 2.5)

    # images = []
    # for i in range(0, HEIGHT_IMAGE_COUNT):
    #     for j in range(0, WIDTH_IMAGE_COUNT):
    #         currentImage = tf.image.crop_to_bounding_box(
    #             image, HEIGHT_INCREMENT * i, WIDTH_INCREMENT * j, IMAGE_DIMENSION, IMAGE_DIMENSION)
    #         currentImage = tf.cast(currentImage, tf.string)
    #         currentImage = tf.io.encode_base64(currentImage)
    #         images.append(currentImage)

    return [prediction]
