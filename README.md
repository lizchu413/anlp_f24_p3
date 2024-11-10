# Cryptic Crossword Solver - Paper Reproduction
cryptic crossword solving with LLMs -- replicating paper results from this paper: https://arxiv.org/pdf/2104.08620

Goal: Numbers that meet or exceeds the previously reported results. A comprehensive analysis of the results, and forward-looking plans for further development.

Note Table 3 in the paper: we attempt to reproduce the ACW + ACW-descramble row

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

```
python train_clues.py --default_train=base --name=naive_top_curricular --project=curricular --wandb_dir='./wandb' --data_dir='./data/clue_json/guardian/naive_random' --multitask=ACW__ACW_descramble --num_workers=1
```

## GETTING THE DATA

Download the data folder from [this link](https://drive.google.com/file/d/1gJLNPqzeCq6uIp_cXujjixKYWT1h9Z09/view?usp=sharing), then unzip and put in this directory (`/anlp_f24_p3`). 