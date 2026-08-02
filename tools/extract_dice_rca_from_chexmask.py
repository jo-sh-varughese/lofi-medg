import pandas as pd

if __name__ == '__main__':
    mimic_cxr_chexmask_path = '../data/chexmask-cxr-segmentation-data/1.0.0/OriginalResolution/MIMIC-CXR-JPG.csv'
    out_path = '../data/mimic/chexmask_mimic_cxr.csv'

    # read csv
    data = pd.read_csv(mimic_cxr_chexmask_path)

    rows = {'dicom_id': [], 'dice_rca_mean': []}
    for _, row in list(data.iterrows()):
        rows['dicom_id'].append(row['dicom_id'])
        rows['dice_rca_mean'].append(row['Dice RCA (Mean)'])

    pd.DataFrame(rows).to_csv(out_path, index=False, encoding='utf-8-sig')
