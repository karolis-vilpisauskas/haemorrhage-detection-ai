import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.optimizers import Adam
import segmentation_models as sm
import tensorflow_advanced_segmentation_models as tasm

# Load the model
current_loss = "dice"
sm.set_framework("tf.keras")
n_classes = 1
BACKBONE = 'efficientnetb7'
activation = 'sigmoid'
model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)
model.compile(optimizer=Adam(lr=1e-4), loss=tasm.losses.DiceLoss(),
              metrics=['accuracy', tasm.metrics.FScore(), tasm.metrics.IOUScore()])
model.load_weights("./unet-b7-weights/" + current_loss + "/best_weights.hdf5")


class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # Define the window properties
        self.title("Image Predictor")
        self.geometry("800x400")

        # Create a label to show the original image
        self.lbl_original = tk.Label(self)
        self.lbl_original.pack(side=tk.LEFT)

        # Create a label to show the predicted image
        self.lbl_predicted = tk.Label(self)
        self.lbl_predicted.pack(side=tk.LEFT)

        # Create a button to upload an image
        self.btn_upload = tk.Button(
            self, text="Upload Image", command=self.upload_image)
        self.btn_upload.pack()

    def upload_image(self):
        # Open a file dialog to select an image file
        filepath = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select a file", filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")])

        if filepath:
            # Load the original image file and resize it to the desired size
            original_img = Image.open(filepath).resize((1280, 1280))

            # Split the image into patches of size 256x256
            patches = []
            for y in range(0, original_img.height, 256):
                for x in range(0, original_img.width, 256):
                    patch = original_img.crop((x, y, x+256, y+256))
                    patches.append(patch)

            # Predict the class of each patch using the model
            predicted_patches = []
            for patch in patches:
                # Convert the patch to a numpy array
                patch_arr = np.array(patch)

                # Normalize the pixel values to be between 0 and 1
                patch_arr = patch_arr / 255.0

                # Add a batch dimension to the array
                patch_arr = np.expand_dims(patch_arr, axis=0)

                # Predict the class of the patch using the model
                prediction = model.predict(patch_arr)
                predicted_class = np.argmax(prediction)

                # Create an image of the predicted class
                color = (0, 191, 255) if predicted_class == 1 else (255, 255, 255)
                predicted_patch = Image.new('RGB', patch.size, color)

                # Add the predicted patch to the list of predicted patches
                predicted_patches.append(predicted_patch)

            # Rebuild the patches back into the full image
            predicted_img = Image.new('RGB', original_img.size, (255, 255, 255))
            patch_idx = 0
            for y in range(0, original_img.height, 256):
                for x in range(0, original_img.width, 256):
                    predicted_patch = predicted_patches[patch_idx]
                    predicted_img.paste(predicted_patch, (x, y))
                    patch_idx += 1

            # Convert the predicted image to RGBA mode
            predicted_img = predicted_img.convert('RGBA')

            # Create a mask image with alpha channel from the black pixels of the predicted image
            mask_data = np.zeros((predicted_img.size[1], predicted_img.size[0], 4), dtype=np.uint8)
            black_pixels = np.all(np.array(predicted_img) == [0, 0, 0, 255], axis=-1)
            mask_data[:, :, 3][black_pixels] = 255
            mask_img = Image.fromarray(mask_data)

            # Create a light blue version of the predicted image
            blue_img = Image.new('RGBA', predicted_img.size, (0, 191, 255, 255))

            # Blend the predicted and blue images using the mask image
            blended_img = Image.composite(blue_img, predicted_img, mask_img)

            # Overlay the augmented image on the original image
            alpha = 0.5
            blended_img = blended_img.resize(original_img.size)
            augmented_img = Image.blend(original_img, blended_img, alpha)

            # Show the original and augmented images on the labels
            original_img_tk = ImageTk.PhotoImage(original_img)
            self.lbl_original.config(image=original_img_tk)
            self.lbl_original.image = original_img_tk

            augmented_img_tk = ImageTk.PhotoImage(augmented_img)
            self.lbl_predicted.config(image=augmented_img_tk)
            self.lbl_predicted.image = augmented_img_tk


if __name__ == '__main__':
    app = App()
    app.mainloop()
