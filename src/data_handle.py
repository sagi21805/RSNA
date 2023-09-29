import numpy as np
import os 
import tensorflow as tf
from src.config import TRAIN_IMG_PATH, IMAGE_SIZE, TARGET_CLASSES, TRAIN_LABEL_PATH, AUTOTUNE, BATCH_SIZE
import pandas as pd
import cv2 
import pydicom as dicom


def decode_dicom(img_path):
    image = dicom.dcmread(img_path).pixel_array
    image = cv2.resize(image, IMAGE_SIZE)
    image = tf.constant(image, tf.float32) / 255.0
    return tf.reshape(image, IMAGE_SIZE + [3, ])

def decode_jpeg(img_path):
    image = cv2.imread(img_path)
    image = cv2.resize(image, IMAGE_SIZE)
    image = tf.constant(image, tf.float32) / 255.0
    return tf.reshape(image, [3, ] + IMAGE_SIZE)
    
def decode_image_and_label(img_path: str, label):
    file_bytes = tf.io.read_file(img_path)
    image = tf.io.decode_jpeg(file_bytes, channels=3)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.cast(label, tf.float32)
    #         bowel       extra      kidney      liver       spleen
    labels = (label[0:1], label[1:2], label[2:5], label[5:8], label[8:11])
    # image = cv2.filter2D(image.numpy(), -1, np.array([[ 0, -1,  0], 
    #                                                   [-1,  5, -1], 
    #                                                   [ 0, -1,  0]]))
    return (image, labels)

def build_dataset(): 
    train_data = pd.read_csv(TRAIN_LABEL_PATH)
    id_label_dict = {label[0]: label[1:-1] for label in train_data[TARGET_CLASSES].values}
    img_paths = []
    labels = []
    for patient_id in os.listdir(TRAIN_IMG_PATH):
        for img_path in os.listdir(TRAIN_IMG_PATH + "/" + patient_id):
            img_paths.append(TRAIN_IMG_PATH + "/" + patient_id + "/" + img_path)
            labels.append(id_label_dict[int(img_paths[-1].split("/")[-2])]) 

    if len(img_paths) != len(labels):
        raise Exception("\n***************\n\
                        img_paths and labels must be in the same length\
                        \n***************\n")  

    ds = tf.data.Dataset.from_tensor_slices((img_paths, labels))\
        .map(decode_image_and_label, num_parallel_calls=AUTOTUNE)\
        .shuffle(BATCH_SIZE * 10)\
        .batch(BATCH_SIZE)\
        .prefetch(AUTOTUNE)\
    
    
    return ds

