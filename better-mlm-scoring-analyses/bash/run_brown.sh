#!/bin/bash
#
#SBATCH --job-name=brown
#SBATCH -e brown-%j.err
#SBATCH -o brown-%j.out
#SBATCH --nodes=1
#SBATCH --mem=100G
#SBATCH -t 20:00:00
#SBATCH -p evlab

timestamp() {
  date +"%T"
}

export DATASET="Brown"
export MODELNAME="bert-base-cased"
timestamp

echo "SOURCING minicons-env ENVIRONMENT (ANACONDA39)"
source /om2/user/ckauf/anaconda39/etc/profile.d/conda.sh
conda activate minicons-env

cd /nese/mit/group/evlab/u/ckauf/minicons/carina
echo "DATASET: " $DATASET
echo "MODEL NAME: " $MODELNAME
echo "MASKING: ${1}"

export PYTHONPATH="${PYTHONPATH}:/nese/mit/group/evlab/u/ckauf/minicons/"
python dataset_scoring.py --dataset $DATASET --model $MODELNAME --which_masking "${1}" --batch_size 200

timestamp
echo 'All complete!'

#for cond in global_l2r ; do sbatch run_brown.sh $cond ; done
#for cond in original within_word_l2r within_word_mlm ; do sbatch run_brown.sh $cond ; done
#for cond in within_word_l2r ; do sbatch run_brown.sh $cond ; done
