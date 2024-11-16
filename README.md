# Cryptic Crossword Solver - Paper Reproduction
cryptic crossword solving with LLMs -- replicating paper results from this paper: https://arxiv.org/pdf/2104.08620

Goal: Numbers that meet or exceeds the previously reported results. A comprehensive analysis of the results, and forward-looking plans for further development.

Note Table 2 in the paper: we attempt to reproduce the very last row (ACW + ACW-descramble)

## Setup Instructions

```
conda create -n decrypt python=3.10 ipython
conda activate decrypt
python -m pip install -r requirements.txt
```

Setting up the datasets: 

```
python -m setup_clue_files
python -m setup_curricular
```

Running the code: 

Baseline WordNet (disjoint) + evaluation (metrics)
```
python -m setup_wordnet
```

Baseline (T5) disjoint (using word-initial disjoint data, with lengths)
```
python train_clues.py --default_train=base --name=baseline_disj --project=baseline --wandb_dir='./wandb' --data_dir='data/clue_json/guardian/word_init_disjoint/'
```

For top result in Table 2 of the paper

```
python train_clues.py --default_train=base --name=naive_top_curricular --project=curricular --wandb_dir='./wandb' --data_dir='data/clue_json/guardian/naive_random/' --multitask=final_top_result_scaled_up
```

## Evaluation

Generating test set predictions using the best checkpoint on baseline (note that `ckpt_path` needs to be replaced with the best performing checkpoint from the command above): 
```
python train_clues.py --default_val=base --name=baseline_disj_val --project=baseline --data_dir='data/clue_json/guardian/word_init_disjoint/' --ckpt_path='./wandb/wandb/run-20241116_184041-kvrnpetp/files/epoch_14.pth.tar' --wandb_dir=./wandb --test
```

Take note of the `..preds.json` file created by this run to compute the metrics: 

```
python -m run_metrics --file "FILEPATHHERE"
```

## Accessing Data (skip dataset setup)

Download the data folder from [this link](https://drive.google.com/file/d/1gJLNPqzeCq6uIp_cXujjixKYWT1h9Z09/view?usp=sharing), then unzip and put in this directory (`/anlp_f24_p3`). 
