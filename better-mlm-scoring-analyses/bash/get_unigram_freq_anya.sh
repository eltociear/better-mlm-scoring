#!/bin/bash
#
#SBATCH --job-name=get_unigram_freq
#SBATCH --output=get_unigram_freq_%j.out
#SBATCH --error=get_unigram_freq_%j.err
#SBATCH -p evlab
#SBATCH --array=0-8
#SBATCH -t 05:00:00

timestamp() {
  date +"%T"
}

export DATASET="Brown"

echo 'Executing get_unigram_freq'
timestamp

cd /nese/mit/group/evlab/u/ckauf/minicons/carina
module load openmind/miniconda/4.0.5-python3

i=0
for chunk in 1 2 3 4 5 6 7 8 9 ; do #0
    chunk_list[$i]="$chunk"
    i=$[$i +1]
done

echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "DATASET: " $DATASET
echo "CHUNK: " ${chunk_list[$SLURM_ARRAY_TASK_ID]}

python get_ngram_frequencies_anya.py --dataset $DATASET --chunk ${chunk_list[$SLURM_ARRAY_TASK_ID]}

timestamp

echo 'All complete!'
