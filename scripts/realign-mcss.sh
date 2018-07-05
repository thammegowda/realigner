#!/usr/bin/env bash
# Author = Thamme Gowda ; Created : July 04, 2018
#$ -cwd -P other -pe mt 18 -l h_vmem=50G,h_rt=24:00:00,gpu=0

# memory requirement depends on vector size and threads
# 100K source words + 300k target words required about 2.5GB RAM per thread (each vector is 300dims )


if [[ $# -ne 1 ]]; then
    echo "ERROR: Usage: <il9 | il10>"
    exit 1
fi
lang=$1

source ~tg/.bashrc
source activate elisa
CODE=/nas/material/users/tg/work/elisa/realiner
export PYTHONPATH=$CODE

FDIR=/nas/material/users/tg/work/elisa/y3-eval/root.y3eval/${lang}/expanded/lrlp/set0/data/translation/found
VECT_DIR=/nas/material/users/tg/work/elisa/rpi

python3 $CODE/realigner.py -fd $FDIR -l $lang -se $VECT_DIR/vectors-$lang.txt -ee $VECT_DIR/vectors-eng.txt --threshold 0.01 --threads 18
