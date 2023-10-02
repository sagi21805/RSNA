import os
import cv2
import pydicom
import tensorflow as tf
import numpy as np
import pandas as pd
from config import IMAGE_SIZE
from glob import glob

np.set_printoptions(suppress=True)

BASE_PATH = "/kaggle/input/rsna-2023-abdominal-trauma-detection"

# IMAGE_DIR = "/tmp/dataset/rsna-atd"

MODEL_PATH = "/home/sagi/Desktop/VsCode/Competiton/MODEL/rsna_model_conv.keras"

STRIDE = 10

TARGET_CLASSES  = ["bowel_healthy", "bowel_injury", 
                   "extravasation_healthy","extravasation_injury", 
                   "kidney_healthy", "kidney_low","kidney_high", 
                   "liver_healthy", "liver_low", "liver_high",
                   "spleen_healthy", "spleen_low", "spleen_high"]


model = tf.keras.models.load_model(MODEL_PATH)
model.summary()

meta_df = pd.read_csv("/home/sagi/Desktop/VsCode/Competiton/RSNA-2023/test_series_meta.csv") #f"{BASE_PATH}/test_series_meta.csv"

meta_df["dicom_folder"] = "/home/sagi/Desktop/VsCode/Competiton/DATA/rsna_test" \
                                    + "/" + meta_df.patient_id.astype(str)\
                                    + "/" + meta_df.series_id.astype(str)
                                    

#! only takes the first dcm.

test_folders = meta_df.dicom_folder.tolist()
test_paths = []
for folder in (test_folders):
    test_paths += sorted(glob(os.path.join(folder, "*dcm")))[::STRIDE]


test_df = pd.DataFrame(test_paths, columns=["dicom_path"])
test_df["patient_id"] = test_df.dicom_path.map(lambda x: x.split("/")[-3]).astype(int)
test_df["series_id"] = test_df.dicom_path.map(lambda x: x.split("/")[-2]).astype(int)
test_df["instance_number"] = test_df.dicom_path.map(lambda x: x.split("/")[-1].replace(".dcm","")).astype(int)


# test_df["image_path"] = f"{IMAGE_DIR}/test_images"\
#                     + "/" + test_df.patient_id.astype(str)\
#                     + "/" + test_df.series_id.astype(str)\
#                     + "/" + test_df.instance_number.astype(str) +".png"
                    
def standardize_pixel_array(dcm):
    # Correct DICOM pixel_array if PixelRepresentation == 1.
    pixel_array = dcm.pixel_array
    if dcm.PixelRepresentation == 1:
        bit_shift = dcm.BitsAllocated - dcm.BitsStored
        dtype = pixel_array.dtype 
        new_array = (pixel_array << bit_shift).astype(dtype) >>  bit_shift
        pixel_array = pydicom.pixel_data_handlers.util.apply_modality_lut(new_array, dcm)
    return pixel_array

def read_xray(path, fix_monochrome=True, batch_size: int = 1):
    dicom = pydicom.dcmread(path)
    data = standardize_pixel_array(dicom)
    data = data - np.min(data)
    data = data / (np.max(data) + 1e-5)
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = 1.0 - data
    data = cv2.resize(data, IMAGE_SIZE)
    data = (data * 255).astype(np.uint8)
    data: np.ndarray = data.reshape(data.shape + (1, ))
    #turn data from 2d array into 3d
    data = np.concatenate((data, data, data), axis = 2)
    # image = tf.cast(data, tf.float32) / 255.0
    data = data.reshape((batch_size, ) + data.shape)
    
    return data

paths  = test_df["dicom_path"].tolist()




def post_proc(pred):
    proc_pred = np.empty((pred.shape[0], 2*2 + 3*3), dtype= np.float32)

    # bowel, extravasation
    proc_pred[:, 0] = pred[:, 0].round(0)
    proc_pred[:, 1] = 1 - proc_pred[:, 0].round(0)
    proc_pred[:, 2] = pred[:, 1].round(0)
    proc_pred[:, 3] = 1 - proc_pred[:, 2].round(0)
    
    # liver, kidney, sneel
    proc_pred[:, 4:7] = pred[:, 2:5].round(0)
    proc_pred[:, 7:10] = pred[:, 5:8].round(0)
    proc_pred[:, 10:13] = pred[:, 8:11].round(0)

    return proc_pred

patient_ids = test_df["patient_id"].unique()

# Initializing array to store predictions
patient_preds = np.zeros(
    shape=(len(patient_ids), 2*2 + 3*3),
    dtype="float32"
)

print(patient_ids)
# Iterating over each patient
for pidx, patient_id in enumerate(patient_ids):
    print(f"Patient ID: {patient_id}")
    
    # Query the dataframe for a particular patient
    patient_df = test_df.iloc[pidx]
    # Getting image paths for a patient
    patient_paths = [patient_df.dicom_path]
    # Building dataset for prediction    
    # Predicting with the model
    sample = read_xray(patient_paths[0])
    pred = model.predict(sample)
    pred = np.concatenate(pred, axis=-1).astype("float32")
    pred = pred[:1, :]
    pred = np.mean(pred.reshape(1, len(patient_paths), 11), axis=0)
    pred = np.max(pred, axis=0, keepdims=True)
    
    patient_preds[pidx, :] += post_proc(pred)[0]
    

    # Deleting variables to free up memory 
    del patient_df, patient_paths, sample, pred
    
pred_df = pd.DataFrame({"patient_id":patient_ids,})
pred_df[TARGET_CLASSES] = patient_preds.astype("float32")

# Align with sample submission
sub_df = pd.read_csv("/home/sagi/Desktop/VsCode/Competiton/RSNA-2023/sample_submission.csv")
sub_df = sub_df[["patient_id"]]
sub_df = sub_df.merge(pred_df, on="patient_id", how="left")

# Store submission
sub_df.to_csv("submission.csv",index=False)

