#!/bin/bash
#
#SBATCH --job-name=calculate_word_LL
#SBATCH --output=calculate_word_LL_%j.out
#SBATCH --error=calculate_word_LL_%j.err
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH -t 03:00:00
#SBATCH -p evlab

timestamp() {
  date +"%T"
}

export MODEL=bert-base-cased
export DATASET=EventsAdapt

echo "MODEL: " $MODEL
echo "MASKING: " ${1}
echo "DATASET: " $DATASET


echo "Calculating word likelihoods"
timestamp

echo "SOURCING minicons-env ENVIRONMENT (ANACONDA39)"
source /om2/user/ckauf/anaconda39/etc/profile.d/conda.sh
conda activate minicons-env

cd /nese/mit/group/evlab/u/ckauf/minicons/carina

export PYTHONPATH="${PYTHONPATH}:/nese/mit/group/evlab/u/ckauf/minicons/"

python calculate-word-likelihood.py --model $MODEL --dataset $DATASET --which_masking "${1}"

timestamp

echo 'All complete!'

#for cond in global_l2r ; do sbatch calculate_word_LL_simple.sh $cond ; done
#for cond in original within_word_l2r within_word_mlm ; do sbatch calculate_word_LL_simple.sh $cond ; done
