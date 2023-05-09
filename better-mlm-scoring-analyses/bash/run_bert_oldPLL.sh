#!/bin/bash
#
#SBATCH --job-name=blimp_bert-cased_oldPLL
#SBATCH --output=blimp_bert-cased_oldPLL_%j.out
#SBATCH --error=blimp_bert-cased_oldPLL_%j.err
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH -t 20:00:00
#SBATCH -p evlab

timestamp() {
  date +"%T"
}

echo ‘Calculating BLiMP scores for bert-base-cased model WITHOUT new metric’
timestamp

filename="blimp_bert-cased_oldPLL_${1}_$(date '+%Y%m%d%T').txt"

echo "SOURCING minicons-env ENVIRONMENT (ANACONDA39)"
source /om2/user/ckauf/anaconda39/etc/profile.d/conda.sh
conda activate minicons-env

cd /nese/mit/group/evlab/u/ckauf/minicons/carina

export PYTHONPATH="${PYTHONPATH}:/nese/mit/group/evlab/u/ckauf/minicons/"

python runBLiMP_experiments.py --model bert-base-cased > "/nese/mit/group/evlab/u/ckauf/minicons/carina/bash/$filename"

timestamp

echo 'All complete!'

