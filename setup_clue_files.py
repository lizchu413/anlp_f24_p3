### CLUE DATASET BASELINE SETUP
# referencing the setup done in https://github.com/jsrozner/decrypt/blob/main/baselines/baseline_t5.ipynb

from decrypt.scrape_parse import (
    load_guardian_splits,
    load_guardian_splits_disjoint,
    load_guardian_splits_disjoint_hash
)

import os
from decrypt import config
from decrypt.common import validation_tools as vt
from decrypt.common.util_data import clue_list_tuple_to_train_split_json
import logging
logging.getLogger(__name__)

k_json_folder = config.DataDirs.Guardian.json_folder

def make_dataset(split_type: str, overwrite=False):
    assert split_type in ['naive_random', 'naive_disjoint', 'word_init_disjoint']
    if split_type == 'naive_random':
        load_fn = load_guardian_splits
        tgt_dir = config.DataDirs.DataExport.guardian_naive_random_split
    elif split_type == 'naive_disjoint':
        load_fn = load_guardian_splits_disjoint
        tgt_dir = config.DataDirs.DataExport.guardian_naive_disjoint_split
    else:
        load_fn = load_guardian_splits_disjoint_hash
        tgt_dir = config.DataDirs.DataExport.guardian_word_init_disjoint_split

    _, _, (train, val, test) = load_fn(k_json_folder)

    os.makedirs(tgt_dir, exist_ok=True)
    # write the output as json
    try:
        clue_list_tuple_to_train_split_json((train, val, test),
                                            comment=f'Guardian data. Split: {split_type}',
                                            export_dir=tgt_dir,
                                            overwrite=overwrite)
    except FileExistsError:
        logging.warning(f'You have already generated the {split_type} dataset.\n'
                        f'It is located at {tgt_dir}\n'
                        f'To regenerate, pass overwrite=True or delete it\n')


make_dataset('naive_random')
make_dataset('word_init_disjoint')
# you can also make_dataset('naive_disjoint')
