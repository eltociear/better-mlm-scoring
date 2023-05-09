#!/bin/bash
#
#SBATCH --job-name=eventsAdapt
#SBATCH --nodes=1
#SBATCH --mem=60G
#SBATCH -t 04:00:00
#SBATCH -p evlab

timestamp() {
  date +"%T"
}

export DATASET="EventsAdapt"
export MODELNAME="gpt2-xl"

echo "MODEL NAME: " $MODELNAME
echo "DATASET: " $DATASET

echo "SOURCING minicons-env ENVIRONMENT (ANACONDA39)"
source /om2/user/ckauf/anaconda39/etc/profile.d/conda.sh
conda activate minicons-env

cd /nese/mit/group/evlab/u/ckauf/minicons/carina
export PYTHONPATH="${PYTHONPATH}:/nese/mit/group/evlab/u/ckauf/minicons/"

python dataset_scoring.py --dataset $DATASET --model $MODELNAME --batch_size 200

timestamp
echo 'All complete!'
