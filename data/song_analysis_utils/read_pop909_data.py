import os
import numpy as np
from tqdm import tqdm
from .read_file import read_data


TRIPLE_METER_SONG = [
    34, 62, 102, 107, 152, 173, 176, 203, 215, 231, 254, 280, 307, 328, 369,
    584, 592, 653, 654, 662, 744, 749, 756, 770, 799, 843, 869, 872, 887
]


DATA_PATH = os.path.join(os.path.dirname(__file__), '../pop909_data')

DATASET_PATH = os.path.join(DATA_PATH, 'pop909_w_structure_label')
ACC_DATASET_PATH = os.path.join(DATA_PATH, 'matched_pop909_acc')

LABEL_SOURCE = np.load(os.path.join(DATA_PATH, 
                                    'pop909_w_structure_label',
                                    'label_source.npy'))

SPLIT_FILE_PATH = os.path.join(DATA_PATH, 'pop909_split', 'split.npz')


def read_pop909_dataset(song_ids=None, label_fns=None, desc_dataset=None):
    """If label_fn is None, use default the selected label file in LABEL_SOURCE"""

    dataset = []

    song_ids = [si for si in range(1, 910)] if song_ids is None else song_ids

    for idx, i in enumerate(tqdm(song_ids, desc=None if desc_dataset is None else f'Loading {desc_dataset}')):
        # which human label file to use
        label = LABEL_SOURCE[i - 1] if label_fns is None else label_fns[idx]

        if i in TRIPLE_METER_SONG:
            continue

        num_beat_per_measure = 3 if i in TRIPLE_METER_SONG else 4

        song_name = str(i).zfill(3)  # e.g., '001'

        data_fn = os.path.join(DATASET_PATH, song_name)  # data folder of the song

        acc_fn = os.path.join(ACC_DATASET_PATH, song_name)

        song_data = read_data(data_fn, acc_fn, num_beat_per_measure=num_beat_per_measure, num_step_per_beat=4,
                              clean_chord_unit=num_beat_per_measure, song_name=song_name, label=label)

        dataset.append(song_data)

    return dataset