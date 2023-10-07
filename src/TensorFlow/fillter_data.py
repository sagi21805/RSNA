# import os
# from PIL import Image
# import imagehash

# def find_duplicates(dirname, hash_size):

#     fnames = os.listdir(dirname)
#     hashes = {}
#     duplicates = []
#     for image in fnames:
#         with Image.open(os.path.join(dirname,image)) as img:
#             temp_hash = imagehash.average_hash(img, hash_size)
#             if temp_hash in hashes:
#                 duplicates.append(image)
#             else:
#                 hashes[temp_hash] = image
                
#     if len(duplicates) != 0:
#         for duplicate in duplicates:            
#             os.remove(os.path.join(dirname,duplicate))
#         print("removed")

# path = r"C:\VsCode\python\machineLearning\machine-learning\competitions\RSNA 2023 Abdominal Trauma Detection\rsna_256x256_jpeg"

# for i, patient_path in enumerate(os.listdir(path)):
#     print(f"[{i/3146 * 100}]" , end = "\r")
#     find_duplicates(path + "\\" + patient_path, 16)



    