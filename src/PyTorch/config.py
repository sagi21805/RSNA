IMAGE_SIZE = [256, 256]

BATCH_SIZE = 256

RESIZE_DIM = 256

EPOCHS = 5

TARGET_CLASSES = [
    
        "patient_id",
        "bowel_injury", "extravasation_injury",
        "kidney_healthy", "kidney_low", "kidney_high",
        "liver_healthy", "liver_low", "liver_high",
        "spleen_healthy", "spleen_low", "spleen_high",
        "any_injury"
    ]


TRAIN_LABEL_PATH = "/home/sagi/Desktop/vscode/RSNA/data-csv/train.csv"  #"/home/sagi/Desktop/VsCode/Competiton/RSNA-2023/train.csv" 

TRAIN_IMG_PATH = "/mnt/c/rsna_256x256_jpeg_filtered/" #"/home/sagi/Desktop/VsCode/Competiton/DATA/rsna_256x256_jpeg_filtered"

TESTING_IMAGES_PATH = r"C:\VsCode\competitions\RSNA 2023 Abdominal Trauma Detection\test_images" #"/kaggle/input/rsna-2023-abdominal-trauma-detection/test_images"

# L2_REGULATOR = tf.keras.regularizers.l2(0.3)
SHUFFLE = True