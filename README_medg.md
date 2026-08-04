# MedG Data

# 🌐 Prepare Datasets

Download the datasets under `./data/`.

### IMed-361M

    git clone https://huggingface.co/datasets/General-Medical-AI/IMed-361M
    cd IMed-361M
    for f in *.zip; do unzip "$f" -d "${f%.zip}"; done

Then run this Python to align the dataset dir structures:

```python
from pathlib import Path
import shutil

root = Path('.')
for d in filter(Path.is_dir, root.iterdir()):
    if (d / 'dataset.json').is_file():
        continue
    print('Move:', d.name)
    sub, = d.iterdir()
    for f in list(sub.iterdir()):
        shutil.move(str(f), str(d))
    sub.rmdir()
```

### Totalsegmentator

    mkdir Totalsegmentator && cd Totalsegmentator
    https://zenodo.org/records/10047292
    https://zenodo.org/records/14710732
    for f in *.zip; do unzip "$f" -d "${f%.zip}"; done

### MIMIC-CXR

    # MIMIC-CXR-JPG
    wget -r -N -c -np --user <USER> --ask-password https://physionet.org/files/mimic-cxr-jpg/2.1.0/

    # LLaVA-Rad
    wget -r -N -c -np --user <USER> --ask-password https://physionet.org/files/llava-rad-mimic-cxr-annotation/1.0.0/

    # CheXmask Database
    wget -r -N -c -np https://physionet.org/files/chexmask-cxr-segmentation-data/1.0.0/

    # MIMIC-Ext-CXR-QBA
    wget -r -N -c -np --user <USER> --ask-password https://physionet.org/files/mimic-ext-cxr-qba/1.0.0/

# 🧾 Required Source Files

    # LLaVA-Rad
    ./data/llava-rad-mimic-cxr-annotations-1.0.0/chat_train_MIMIC_CXR_all_gpt4extract_rulebased_v1.json
    ./data/llava-rad-mimic-cxr-annotations-1.0.0/chat_dev_MIMIC_CXR_all_gpt4extract_rulebased_v1.json
    ./data/llava-rad-mimic-cxr-annotations-1.0.0/chat_test_MIMIC_CXR_all_gpt4extract_rulebased_v1.json

# 🗂️ Expected Preprocessing Inputs

    # IMed-361M
    ./data/IMed-361M/

    # Totalsegmentator
    ./data/Totalsegmentator/Totalsegmentator_dataset_v201/
    ./data/Totalsegmentator/TotalsegmentatorMRI_dataset_v200/

    # MIMIC-CXR-JPG
    ./data/mimic-cxr-jpg/2.1.0/files/

    # CheXmask Database
    ./data/chexmask-cxr-segmentation-data/1.0.0/OriginalResolution/MIMIC-CXR-JPG.csv

    # MIMIC-Ext-CXR-QBA
    ./data/mimic-ext-cxr-qba/1.0.0/exports/A_frontal/qa/

# ⚙️ Preprocessing

Run preprocessing scripts from `tools/`.

### IMed-361M and Totalsegmentator

Preprocess IMed-361M and Totalsegmentator for MedG:

    python preprocess_imed.py
    python preprocess_totalseg.py
    python create_medg_mini_test.py  # optional

Expected MedG structure

    .
    ├── train
    │   ├── dataset1
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
    │   ├── dataset2
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
    │   └── ...
    ├── test
    │   ├── dataset1
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
    │   ├── dataset2
    │   │   ├── img1.jpg
    │   │   ├── img2.jpg
    │   │   └── ...
    │   └── ...
    └── json
        ├── train_dataset1.json
        ├── train_dataset2.json
        ├── ...
        ├── test_dataset1.json
        ├── test_dataset2.json
        └── ...

### MIMIC-CXR

Extract CheXmask RCA scores for MIMIC filtering:

    python extract_dice_rca_from_chexmask.py

Resize MIMIC-CXR images:

    python resize_images.py \
        --data_dir ../data/mimic-cxr-jpg/2.1.0/files/ \
        --save_dir ../data/mimic_512p_good/

Save original image sizes for MIMIC-CXR-JPG:

    python save_ori_size.py mimic

Preprocess MIMIC-Ext-CXR-QBA annotations:

    python preprocess_mimic_ext.py --out_path ../data/mimic/mimic_ext.csv

# 📦 Preprocessed Files

    ./data/MedG_512p/
    ./data/mimic_512p_good/

    ./data/mimic/chexmask_mimic_cxr.csv
    ./data/mimic/mimic_ext.csv
    ./data/mimic_ori_size.csv
