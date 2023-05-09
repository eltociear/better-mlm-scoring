#!/bin/bash
#
#SBATCH --job-name=roberta-array
#SBATCH --array=0-60
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH -t 03:00:00
#SBATCH -p evlab

timestamp() {
  date +"%T"
}

export MASKING="original"
export MODEL="bert-base-cased"
timestamp

echo "SOURCING minicons-env ENVIRONMENT (ANACONDA39)"
source /om2/user/ckauf/anaconda39/etc/profile.d/conda.sh
conda activate minicons-env

cd /nese/mit/group/evlab/u/ckauf/minicons/carina

exclude_list=(existential_there_object_raising.jsonl expletive_it_object_raising.jsonl complex_NP_island.jsonl ellipsis_n_bar_2.jsonl ellipsis_n_bar_1.jsonl wh_questions_subject_gap_long_distance.jsonl sentential_negation_npi_scope.jsonl)

i=0
for file in blimp/data/* ; do
    # Check if the file is in the exclude list
    if [[ ! " ${exclude_list[@]} " =~ " ${file##*/} " ]]; then
        for model in $MODEL ; do
            #echo "Scoring pairs in ${file} using model ${model}..."
            mkdir -p output/${model}/
            model_list[$i]="$model"
            file_list[$i]="$file"
            i=$[$i +1]
        done
    fi
done

echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "FILE NAME: " ${file_list[$SLURM_ARRAY_TASK_ID]}
echo "MODEL NAME: " ${model_list[$SLURM_ARRAY_TASK_ID]}
echo "MASKING: " $MASKING

export PYTHONPATH="${PYTHONPATH}:/nese/mit/group/evlab/u/ckauf/minicons/"
python run_BLiMP_exp_individually.py --model ${model_list[$SLURM_ARRAY_TASK_ID]} --file ${file_list[$SLURM_ARRAY_TASK_ID]} --batch_size 500 --which_masking $MASKING

timestamp
echo 'All complete!'
