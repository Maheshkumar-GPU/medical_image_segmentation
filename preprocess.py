import os
import numpy as np
import cv2
import PIL.Image as Image

#  config
DATASET_PATH = r"C:/Users/Asus/OneDrive/Documents/ml project datasets/Brain_MRI/kaggle_3m"
IMG_SIZE = 128
images = []
masks = []

# loop through patients
patients = os.listdir(DATASET_PATH)
for patient in patients:
    patient_path = os.path.join(DATASET_PATH, patient)
    if not os.path.isdir(os.path.join(DATASET_PATH, patient)):
        continue
    files = os.listdir(patient_path)
    image_file = None
    mask_file = None
    for f in files:
        if f.endswith(".tif"):
            if "mask" in f.lower():
                mask_file =f
            else:
                image_file = f
    if image_file is  None or mask_file is None:
        continue

# read image and mask

image_path = os.path.join(patient_path, image_file)
mask_path = os.path.join(patient_path, mask_file)
image = np.array(Image.open(image_path))
mask = np.array(Image.open(mask_path))

# resize
image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
mask = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))

# normalize image
image = image/255.0
# binary mask
mask = np.where(mask > 0, 1, 0)
images.append(image)
masks.append(mask)

# convert to numpy
images = np.array(images)
masks = np.array(masks)
masks = np.expand_dims(masks, axis=-1)
print("image shape ",image.shape)
print("mask shape",mask.shape)
