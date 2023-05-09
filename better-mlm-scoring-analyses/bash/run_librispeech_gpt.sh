#!/bin/bash
#
#SBATCH --job-name=librispeech
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH -t 10:00:00
#SBATCH -p evlab

timestamp() {
  date +"%T"
}

export DATASET="LibriSpeech"
export MODELNAME="gpt2-medium"
timestamp

echo "SOURCING minicons-env ENVIRONMENT (ANACONDA39)"
source /om2/user/ckauf/anaconda39/etc/profile.d/conda.sh
conda activate minicons-env

cd /nese/mit/group/evlab/u/ckauf/minicons/carina
echo "MODEL NAME: " $MODELNAME

export PYTHONPATH="${PYTHONPATH}:/nese/mit/group/evlab/u/ckauf/minicons/"
python dataset_scoring.py --dataset $DATASET --model $MODELNAME --batch_size 200

timestamp
echo 'All complete!'

