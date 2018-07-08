#!/usr/bin/env bash
# Author = Thamme Gowda ; Created : July 08, 2018
#$ -cwd -P other -pe mt 40 -l h_vmem=50G,h_rt=24:00:00,gpu=0

# memory requirement depends on threads
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
TTAB_FILE=/nas/material/users/tg/work/elisa/y3-eval/ttabs/${lang}-eng.lc.ttab.pkl

FLAGS="charlen,toklen,copypatn,ascii,ttab"
python3 $CODE/realigner.py -f ${FLAGS} -fd ${FDIR} -l ${lang} -tf ${TTAB_FILE} \
   --threshold 0.2 --threads 40 -o 'sentence_alignment-ttab'
