#!/bin/bash
#
#SBATCH --job-name=blimp_gpt2-array
#SBATCH --array=0-66
#SBATCH --nodes=1
#SBATCH --mem=20G
#SBATCH -t 01:00:00
#SBATCH -p evlab

timestamp() {
  date +"%T"
}

echo "Calculating BLiMP scores for gpt2"
timestamp

echo "SOURCING minicons-env ENVIRONMENT (ANACONDA39)"
source /om2/user/ckauf/anaconda39/etc/profile.d/conda.sh
conda activate minicons-env

cd /nese/mit/group/evlab/u/ckauf/minicons/carina

i=0
for file in blimp/data/* ; do
    for model in gpt2-medium ; do
        #echo "Scoring pairs in ${file} using model ${model}..."
        mkdir -p output/${model}/
        model_list[$i]="$model"
        file_list[$i]="$file"
        i=$[$i +1]
    done
done

echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "FILE NAME: " ${file_list[$SLURM_ARRAY_TASK_ID]}
echo "MODEL NAME: " ${model_list[$SLURM_ARRAY_TASK_ID]}

export FILENAME="/nese/mit/group/evlab/u/ckauf/minicons/carina/bash/${model_list[$SLURM_ARRAY_TASK_ID]}_$SLURM_ARRAY_TASK_ID.txt"

export PYTHONPATH="${PYTHONPATH}:/nese/mit/group/evlab/u/ckauf/minicons/"
python run_BLiMP_exp_individually.py --model ${model_list[$SLURM_ARRAY_TASK_ID]} --file ${file_list[$SLURM_ARRAY_TASK_ID]} --batch_size 500 > $FILENAME
        
timestamp
echo 'All complete!'
