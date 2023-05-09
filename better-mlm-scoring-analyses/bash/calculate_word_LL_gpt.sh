#!/bin/bash
#
#SBATCH --job-name=brown_calculate_word_LL_gpt
#SBATCH --output=brown_calculate_word_LL_gpt_%j.out
#SBATCH --error=brown_calculate_word_LL_gpt_%j.err
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH -t 05:00:00
#SBATCH --array=0-9
#SBATCH -p evlab

timestamp() {
  date +"%T"
}

export MODEL=gpt2-medium
export DATASET=Brown

i=0
for chunk in 0 1 2 3 4 5 6 7 8 9 ; do
    chunk_list[$i]="$chunk"
    i=$[$i +1]
done

echo "MODEL: " $MODEL
echo "DATASET: " $DATASET
echo "CHUNK: " ${chunk_list[$SLURM_ARRAY_TASK_ID]}

echo "Calculating word likelihoods"
timestamp

echo "SOURCING minicons-env ENVIRONMENT (ANACONDA39)"
source /om2/user/ckauf/anaconda39/etc/profile.d/conda.sh
conda activate minicons-env

cd /nese/mit/group/evlab/u/ckauf/minicons/carina

export PYTHONPATH="${PYTHONPATH}:/nese/mit/group/evlab/u/ckauf/minicons/"

python calculate-word-likelihood.py --model $MODEL --chunk ${chunk_list[$SLURM_ARRAY_TASK_ID]} --dataset $DATASET

timestamp

echo 'All complete!'
