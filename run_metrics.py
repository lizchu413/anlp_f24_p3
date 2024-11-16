### PRODUCE METRICS BASED ON FILE NAME
from decrypt.scrape_parse import (
    load_guardian_splits,
    load_guardian_splits_disjoint,
    load_guardian_splits_disjoint_hash
)

import os
import argparse
from decrypt import config
from decrypt.common import validation_tools as vt
from decrypt.common.util_data import clue_list_tuple_to_train_split_json
from decrypt.common.validation_tools import load_and_run_t5
import logging
logging.getLogger(__name__)

# load_and_run_t5('preds_out/baseline_disj_val')


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-file', type = str,
                        help = "prediction file name (should be a json", 
                       default = 'preds_out/baseline_disj_val.json')
    return parser

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    if '.json' in args.file: 
        args.file = args.file.split('.json')[0]
    load_and_run_t5(args.file)
