#!/bin/bash
#
#SBATCH --job-name=librispeech
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH -t 10:00:00
#SBATCH -p evlab

timestamp() {
  date +"%T"
}

export DATASET="LibriSpeech"
export MODELNAME="bert-base-cased"
export MASKING="global_l2r"
timestamp

echo "SOURCING minicons-env ENVIRONMENT (ANACONDA39)"
source /om2/user/ckauf/anaconda39/etc/profile.d/conda.sh
conda activate minicons-env

cd /nese/mit/group/evlab/u/ckauf/minicons/carina
echo "MODEL NAME: " $MODELNAME
echo "MASKING: " $MASKING

export PYTHONPATH="${PYTHONPATH}:/nese/mit/group/evlab/u/ckauf/minicons/"
python dataset_scoring.py --dataset $DATASET --model $MODELNAME --which_masking $MASKING --batch_size 200
#python dataset_scoring.py --dataset $DATASET --model $MODELNAME --batch_size 200

timestamp
echo 'All complete!'

