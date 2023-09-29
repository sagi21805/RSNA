import pandas as pd
import os 
import numpy as np
from src.config import TESTING_IMAGES_PATH, TRAIN_IMG_PATH
import tensorflow as tf
from src.data_handle import decode_dicom, decode_jpeg
import cv2

TESTING_IMAGES_PATH = TRAIN_IMG_PATH

np.set_printoptions(suppress=True)

model = tf.keras.models.load_model("/home/sagi21805/Desktop/VsCode/competitions/RSNA-2023-Abdominal-Trauma-Detection/test_model.keras")

model.summary()

samp_sub_df = pd.read_csv("sample_submission.csv")
header = samp_sub_df.columns.values.tolist()
empty = {        
            "patient_id" : [0],
            "bowel_healthy": [0],
            "bowel_injury" : [0], 
            "extravasation_healthy" : [0], 
            "extravasation_injury" : [0], 
            "kidney_healthy" : [0],
            "kidney_low" : [0],
            "kidדney_high" : [0],
            "liver_healthy" : [0] ,
            "liver_low" : [0],
            "liver_high" : [0],
            "spleen_healthy"  : [0],
            "spleen_low" : [0],
            "spleen_high" : [0]  
            }
sub_df = pd.DataFrame(columns=header)
for patient_id in os.listdir(TESTING_IMAGES_PATH):
    data = pd.DataFrame(empty)
    reps = 0
    # for series_id in os.listdir(TESTING_IMAGES_PATH + "/" + patient_id):
    for img_path in os.listdir(TESTING_IMAGES_PATH + "/" + patient_id):
        image = decode_jpeg(TESTING_IMAGES_PATH + "/" + patient_id + "/" + img_path)
        bowel_injury, extra_injury, kidney, liver, spleen = model.predict(image, verbose='silent')
        
        arr = np.array([1-bowel_injury[0][0], bowel_injury[0][0]])
        bowel_healty, bowel_injury = np.vectorize(lambda x: int(x >= 1/len(arr)))(arr)
        
        arr = np.array([1-extra_injury[0][0], extra_injury[0][0]])
        extra_healty, extra_injury = np.vectorize(lambda x: int(x >= 1/len(arr)))(arr)

        arr = kidney[0]
        kidney_healthy, kidney_low, kidney_high = np.vectorize(lambda x: int(x >= 1/len(arr)))(arr)

        arr = liver[0]
        liver_healthy, liver_low, liver_high = np.vectorize(lambda x: int(x >= 1/len(arr)))(arr)

        arr = spleen[0]
        spleen_healthy, spleen_low, spleen_high = np.vectorize(lambda x: int(x >= 1/len(arr)))(arr)

        data_d = {        
        "patient_id" : int(patient_id),
        "bowel_healthy": [bowel_healty],
        "bowel_injury" : [bowel_injury], 
        "extravasation_healthy" : [extra_healty], 
        "extravasation_injury" : [extra_injury], 
        "kidney_healthy" : [kidney_healthy],
        "kidney_low" : [kidney_low],
        "kidney_high" : [kidney_high],
        "liver_healthy" :  [liver_healthy],
        "liver_low" : [liver_low],
        "liver_high" : [liver_high],
        "spleen_healthy"  : [spleen_healthy],
        "spleen_low" : [spleen_low],
        "spleen_high" : [spleen_high]
        }
        data += pd.DataFrame(data_d)
        reps += 1
        break
    data = data / reps
    data = data.astype({'patient_id':'int'})
    sub_df = pd.concat([sub_df, data])

sub_df = sub_df.sort_values(by=['patient_id'])
sub_df.to_csv("submission.csv", index = False)


