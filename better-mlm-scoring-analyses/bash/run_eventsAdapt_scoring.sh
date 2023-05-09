#!/bin/bash
#
#SBATCH --job-name=eventsAdapt
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH -t 03:00:00
#SBATCH -p evlab

timestamp() {
  date +"%T"
}

export DATASET="EventsAdapt"
export MODELNAME="bert-base-cased"
export MASKING="${1}"

echo "MODEL NAME: " $MODELNAME
echo "DATASET: " $DATASET
echo "MASKING: " $MASKING

echo "SOURCING minicons-env ENVIRONMENT (ANACONDA39)"
source /om2/user/ckauf/anaconda39/etc/profile.d/conda.sh
conda activate minicons-env

cd /nese/mit/group/evlab/u/ckauf/minicons/carina
export PYTHONPATH="${PYTHONPATH}:/nese/mit/group/evlab/u/ckauf/minicons/"

python dataset_scoring.py --dataset $DATASET --model $MODELNAME --which_masking $MASKING --batch_size 200

timestamp
echo 'All complete!'

# for cond in "global_l2r" ; do sbatch run_eventsAdapt_scoring.sh $cond ; done

# for cond in "original" "within_word_mlm" "within_word_l2r" ; do sbatch run_eventsAdapt_scoring.sh $cond ; done
