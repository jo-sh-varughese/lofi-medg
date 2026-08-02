# Downstream Data

# Prepare Datasets

Download the datasets under `./data/`.

    # PadChest-GR
    https://bimcv.cipf.es/bimcv-projects/padchest-gr/

    # TN5000
    https://figshare.com/s/cb6a67f17c04b29e7edd

    # SegTHOR
    https://codalab.lisn.upsaclay.fr/competitions/843

    # SLAKE
    git clone https://huggingface.co/datasets/BoKelvin/SLAKE
    cd SLAKE
    unzip imgs.zip

    # VQA-RAD
    https://osf.io/89kps/overview
    https://github.com/Google-Health/google-health/blob/master/data_splits/vqa_rad_balanced_split_and_human_eval_inclusions.tsv

    # OmniMedVQA
    https://huggingface.co/datasets/foreverbeliever/OmniMedVQA

# Required Source Files
    
    ./data/SLAKE/
    ./data/VQA_RAD/

    ./data/vqarad/vqa_rad_balanced_split_and_human_eval_inclusions.tsv

    # PadChest-GR
    ./data/padchest/grounded_reports_20240819.json
    ./data/padchest/master_table.csv
    ./data/padchest/ori_size.csv

# Expected Preprocessing Inputs

    ./data/BIMCV-Padchest-GR/Padchest_GR_files/
    ./data/TN5000/
    ./data/SegTHOR/train/
    ./data/OmniMedVQA/

# Preprocessing

Run preprocessing scripts from `tools/`.

    python preprocess_tn5000.py
    python preprocess_segthor.py
    python preprocess_omnimedvqa_cares.py

    # PadChest-GR
    python resize_images.py \
        --data_dir ../data/BIMCV-Padchest-GR/Padchest_GR_files/ \
        --save_dir ../data/padchest_512p/
    python save_ori_size.py padchest

# Preprocessed Files

    ./data/padchest_512p/
    ./data/tn5000_512p/
    ./data/segthor_512p/
    ./data/omnimedvqa_512p/

    ./data/padchest_512p/ori_size.csv

