from config import BATCH_SIZE, EPOCHS
import os
from data_handle import decode_dicom, build_dataset
from model import build_model
import time

aPath = '--xla_gpu_cuda_data_dir=/home/sagi/miniconda3/envs/tf/lib' #path to the directory of libdevice.10.
print(aPath)
os.environ['XLA_FLAGS'] = aPath



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





