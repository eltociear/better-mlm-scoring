#!/bin/bash
#
#SBATCH --job-name=roberta-array_highmem
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH -t 10:00:00
#SBATCH -p evlab

timestamp() {
  date +"%T"
}

export MODELNAME="roberta-large"
timestamp

echo "SOURCING minicons-env ENVIRONMENT (ANACONDA39)"
source /om2/user/ckauf/anaconda39/etc/profile.d/conda.sh
conda activate minicons-env

cd /nese/mit/group/evlab/u/ckauf/minicons/carina

i=0
for file in existential_there_object_raising expletive_it_object_raising complex_NP_island ellipsis_n_bar_2 ellipsis_n_bar_1 wh_questions_subject_gap_long_distance sentential_negation_npi_scope ; do
#for file in ellipsis_n_bar_2 ; do
    for masking in original global_l2r ; do #within_word_l2r within_word_mlm ; do
    mkdir -p output/${MODELNAME}/
    file_list[$i]="blimp/data/${file}.jsonl"
    masking_list[$i]="$masking"
    i=$[$i +1]
    done
done

echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "FILE NAME: " ${file_list[$SLURM_ARRAY_TASK_ID]}
echo "MASKING NAME: " ${masking_list[$SLURM_ARRAY_TASK_ID]}
echo "MODEL NAME: " $MODELNAME

export PYTHONPATH="${PYTHONPATH}:/nese/mit/group/evlab/u/ckauf/minicons/"
python run_BLiMP_exp_individually.py --model $MODELNAME --file ${file_list[$SLURM_ARRAY_TASK_ID]} --which_masking ${masking_list[$SLURM_ARRAY_TASK_ID]} --batch_size 500

timestamp
echo 'All complete!'

