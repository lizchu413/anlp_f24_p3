### WORDNET BASELINE SETUP + RUN
# referencing the setup done in https://github.com/jsrozner/decrypt/blob/main/baselines/baseline_wn.ipynb

from decrypt.scrape_parse import (
    load_guardian_splits,
    load_guardian_splits_disjoint_hash
)

import random
from typing import *
from decrypt import config

import jellyfish

from multiset import Multiset
import nltk
nltk.download('wordnet')

from nltk.corpus import wordnet as wn
from tqdm import tqdm


from decrypt.common.puzzle_clue import GuardianClue
from decrypt.common.util_wordnet import all_inflect
from decrypt.common import validation_tools as vt

# Wordnet functions to produce reverse dictionary sets

k_json_folder = config.DataDirs.Guardian.json_folder

def normalize(lemma):
    """Wordnet returns words with underscores and hyphens. We replace them with spaces. This possibly does not work well with lemminflect."""
    return lemma.replace("_"," ").replace("-"," ")

def get_syns(w: str) -> Set[str]:
    """
    Get all synonyms of w
    """
    ret = set()
    for ss in wn.synsets(w):
        for l in ss.lemma_names():
            ret.add(normalize(l))
    return ret

def get_syns_hypo1(w: str) -> Set[str]:
    """
    Get all synonyms and hyponyms to depth 1
    """
    ret = set()
    for ss in wn.synsets(w):
        for l in ss.lemma_names():
            ret.add(normalize(l))
        for rel_ss in ss.hyponyms():
            for l in rel_ss.lemma_names():
                ret.add(normalize(l))
    return ret

def get_syns_hypo_all(w: str, include_hyper=False, depth=3) -> Set[str]:
    """
    Get all synonyms; hyponyms to depth, depth; and hypernyms to depth, depth,
    if include_hyper is True

    :param w: word to lookup
    :param include_hyper: whether to do hypernym lookup
    :param depth: how far to go in hyponym / hypernym traversal
    """
    ret = set()
    for ss in wn.synsets(w):
        for l in ss.lemma_names():
            ret.add(normalize(l))
        if include_hyper:
            for rel_ss in ss.closure(lambda s: s.hypernyms(), depth=depth):
                for l in rel_ss.lemma_names():
                    ret.add(normalize(l))
        for rel_ss in ss.closure(lambda s: s.hyponyms(), depth=depth):
            for l in rel_ss.lemma_names():
                ret.add(normalize(l))
    return ret

def get_first_and_last_word(c: GuardianClue):
    clue_words = c.clue.split(" ")
    return clue_words[0], clue_words[-1]

def pct_sim(str1, str2):
    max_len = max(len(str1), len(str2))
    lev = jellyfish.levenshtein_distance(str1, str2)
    return 1.0 - lev/max_len

def eval_wn(val_set: List[GuardianClue],
            fcn: Callable,
            do_fuzzy: bool,
            do_rank: bool = False,
            **fcn_kwargs):
    """
    :param val_set:
    :param fcn:
    :param do_fuzzy:
    :param fcn_kwargs:
    :return:
    """
    rng = random.Random()
    rng.seed(42)

    model_outputs = []
    for val_gc in tqdm(val_set):
        all_possible = set()

        # add the direct synonyms
        for w in get_first_and_last_word(val_gc):
            all_possible.update(list(fcn(w.lower(), **fcn_kwargs)))

        # potentially add lemmas
        if do_fuzzy:
            orig = all_possible.copy()
            for w in orig:
                all_possible.update(all_inflect(w, None))

        _, filtered = vt.filter_to_len(val_gc.soln_with_spaces, all_possible)
        filtered_final = [x[0] for x in filtered]   # go back to with spaces

        # jellyfish score
        # # if do_rank:
        # #     list_with_rank = []
        # #     for out in filtered_final:
        # #         score = pct_sim(out, val_gc.clue)
        # #         list_with_rank.append((out, score))
        # #     # sort
        # #     list_sorted = sorted(list_with_rank, key=lambda x: x[1], reverse=True)
        # #     # take the word not the score
        #     filtered_final = [x[0] for x in list_sorted]

        # simple character overlap
        if do_rank:
            list_with_rank = []
            mset = Multiset(val_gc.clue)
            for out in filtered_final:
                score = len(mset.intersection(Multiset(out)))
                list_with_rank.append((out, score))
            # sort
            list_sorted = sorted(list_with_rank, key=lambda x: x[1], reverse=True)
            # take the word not the score
            filtered_final = [x[0] for x in list_sorted]
        else:
            rng.shuffle(filtered_final)

        mp = vt.ModelPrediction(
            idx=val_gc.idx,
            input=val_gc.clue_with_lengths(),
            target=val_gc.soln_with_spaces,
            greedy="",
            sampled=filtered_final)

        mp.model_eval = vt.eval(mp)
        model_outputs.append(mp)

    return model_outputs

##
# run on disjoint set
##

def run_primary_wn_disj2():
    _, _, (_, val_orig, test_orig) = load_guardian_splits_disjoint_hash(k_json_folder)
    print('val results')
    out1 = eval_wn(val_orig, fcn=get_syns_hypo1, do_fuzzy=False, do_rank=True) # 1711
    vt.all_aggregate(out1, label='syns,hypo1; no fuzzy, ranked by char overlap')

    print('test results')
    out2 = eval_wn(test_orig, fcn=get_syns_hypo1, do_fuzzy=False, do_rank=True) # 1711
    vt.all_aggregate(out2, label='syns,hypo1; no fuzzy, ranked by char overlap')

run_primary_wn_disj2()
