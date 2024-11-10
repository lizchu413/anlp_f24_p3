### CURRICULAR EXPERIMENT SETUP
# referencing the setup done in https://github.com/jsrozner/decrypt/blob/main/experiments/curricular.ipynb

from decrypt.scrape_parse.acw_load import get_clean_xd_clues
from decrypt import config
from decrypt.common.util_data import clue_list_tuple_to_train_split_json
from decrypt.common import validation_tools as vt

k_xd_orig_tsv = config.DataDirs.OriginalData.k_xd_cw        # ./data/original/xd/clues.tsv
k_acw_export_dir = config.DataDirs.DataExport.xd_cw_json

# defaults to strip periods, remove questions, remove abbrevs, remove fillin
stc_map, all_clues = get_clean_xd_clues(k_xd_orig_tsv,
                                        remove_if_not_in_dict=False,
                                        do_filter_dupes=True)
clue_list_tuple_to_train_split_json((all_clues,),
                                    comment='ACW set; xd cw set, all',
                                    export_dir=k_acw_export_dir,
                                    overwrite=False)

# produce anagram datasets
# roughly 3 minutes to complete
from decrypt.common import anagrammer
anagrammer.gen_db_with_both_inputs(update_flag="overwrite")

from decrypt.common.util_data import (
    get_anags,
    write_json_tuple
)
import json
import os

def make_anag_sets_json():
    all_anags = get_anags(max_num_words=-1)
    json_list = []
    for idx, a_list in enumerate(all_anags):
        json_list.append(dict(idx=idx,
                              anag_list=a_list))
    print(json_list[0])

    # normally would be (idx, input, tgt)
    output_tuple = [json_list,]

    os.makedirs(config.DataDirs.DataExport.anag_dir)
    write_json_tuple(output_tuple,
                     comment="List of all anagram groupings",
                     export_dir=config.DataDirs.DataExport.anag_dir,
                     overwrite=False)

def make_anag_indic_list_json():
    # make the indicator list
    with open(config.DataDirs.OriginalData.k_deits_anagram_list, 'r') as f:
        all_anag_indicators = f.readlines()
        print(len(all_anag_indicators))

    final_indic_list = []
    for a in all_anag_indicators:
        final_indic_list.append(a.replace('_', " ").strip())
    with open(config.DataDirs.DataExport.anag_indics, 'w') as f:
        json.dump(final_indic_list,f)

make_anag_sets_json()
make_anag_indic_list_json()
