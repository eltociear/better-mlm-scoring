#!/bin/bash
#
#SBATCH --job-name=blimp_gpt2_blimp
#SBATCH --output=blimp_gpt2_%j.out
#SBATCH --error=blimp_gpt2_%j.err
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH -t 20:00:00
#SBATCH -p evlab

timestamp() {
  date +"%T"
}

echo ‘Calculating BLiMP scores for gpt2’
timestamp

filename="blimp_gpt2_${1}_$(date '+%Y%m%d%T').txt"

echo "SOURCING minicons-env ENVIRONMENT (ANACONDA39)"
source /om2/user/ckauf/anaconda39/etc/profile.d/conda.sh
conda activate minicons-env

cd /nese/mit/group/evlab/u/ckauf/minicons/carina

export PYTHONPATH="${PYTHONPATH}:/nese/mit/group/evlab/u/ckauf/minicons/"

python runBLiMP_experiments.py --model gpt2 > "/nese/mit/group/evlab/u/ckauf/minicons/carina/bash/$filename"

timestamp

echo 'All complete!'

