import json
import os
import random

from tqdm import tqdm


def create_mini_test(medg_json_dir, save_json_path, mini_test_size, test_prefix):
    save_test_dict = {'data': {}, 'modality': ''}
    for filename in tqdm(sorted(os.listdir(medg_json_dir))):
        if not filename.endswith('.json'):
            continue

        if not filename.startswith(test_prefix):
            continue

        # read json
        with open(os.path.join(medg_json_dir, filename), 'r', encoding='utf-8') as f:
            json_data = json.load(f)

        data_list = json_data['data']

        # select
        keys = list(data_list.keys())
        random.Random(0).shuffle(keys)
        keys = keys[:mini_test_size]

        if len(keys) != mini_test_size:
            print('check:', filename, len(keys))

        # set
        for key in keys:
            save_test_dict['data'][key] = data_list[key]

    # save json
    with open(save_json_path, 'wt') as f:
        json.dump(save_test_dict, f, indent=4)


if __name__ == '__main__':
    medg_dir = '../data/MedG_512p/'
    medg_json_dir = os.path.join(medg_dir, 'json')
    create_mini_test(
        medg_json_dir=medg_json_dir,
        save_json_path=os.path.join(medg_json_dir, 'minitest_medg.json'),
        mini_test_size=40,
        test_prefix='test_',
    )

    create_mini_test(
        medg_json_dir=medg_json_dir,
        save_json_path=os.path.join(medg_json_dir, 'minitest_Totalsegmentator_medg.json'),
        mini_test_size=500,
        test_prefix='test_Totalsegmentator',
    )
