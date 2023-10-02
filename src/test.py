import numpy as np


import tensorflow as tf

print(tf.__version__)

model = tf.keras.models.load_model("/home/sagi/Desktop/VsCode/Competiton/MODEL/test_model.keras")

print(model.summary())