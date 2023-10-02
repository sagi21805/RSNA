import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.set_logical_device_configuration(
        gpus[0],
        [tf.config.LogicalDeviceConfiguration(memory_limit=3500)])
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

from config import BATCH_SIZE, EPOCHS
import os
from data_handle import decode_dicom, build_dataset
from model import build_model
import time 

print(f"\n\n\n{tf.__version__}\n\n\n")



st = time.time()
print("\n building model")
model = build_model()
print(f"[TIME]: {time.time() - st}")
st = time.time()
print("\n building dataset")
data = build_dataset()
print(f"[TIME]: {time.time() - st}")


model.fit(data, batch_size=BATCH_SIZE, epochs=EPOCHS)

model.save("rsna_model.keras")





