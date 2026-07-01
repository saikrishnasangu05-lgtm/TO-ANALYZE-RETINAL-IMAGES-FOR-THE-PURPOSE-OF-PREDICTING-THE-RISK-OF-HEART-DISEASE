import os
import shutil

source_folder = "dataset_raw/images"

normal_folder = "dataset/normal"
risk_folder = "dataset/risk"

os.makedirs(normal_folder, exist_ok=True)
os.makedirs(risk_folder, exist_ok=True)

for file in os.listdir(source_folder):

    src = os.path.join(source_folder, file)

    if "_h" in file:
        shutil.copy(src, normal_folder)

    elif "_dr" in file or "_g" in file:
        shutil.copy(src, risk_folder)

print("Dataset prepared successfully")