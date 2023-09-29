from src.config import BATCH_SIZE, EPOCHS
import os
from src.data_handle import decode_dicom, build_dataset
from src.model import build_model
import time 


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





