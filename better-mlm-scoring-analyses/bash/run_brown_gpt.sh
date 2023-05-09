#!/bin/bash
#
#SBATCH --job-name=brown-xl
#SBATCH -e brown-gpt-xl-%j.err
#SBATCH -o brown-gpt-xl-%j.out
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH -t 20:00:00
#SBATCH -p evlab

timestamp() {
  date +"%T"
}

export DATASET="Brown"
export MODELNAME="gpt2-xl"
timestamp

echo "SOURCING minicons-env ENVIRONMENT (ANACONDA39)"
source /om2/user/ckauf/anaconda39/etc/profile.d/conda.sh
conda activate minicons-env

cd /nese/mit/group/evlab/u/ckauf/minicons/carina
echo "DATASET: " $DATASET
echo "MODEL NAME: " $MODELNAME

export PYTHONPATH="${PYTHONPATH}:/nese/mit/group/evlab/u/ckauf/minicons/"
python dataset_scoring.py --dataset $DATASET --model $MODELNAME --batch_size 200

timestamp
echo 'All complete!'
