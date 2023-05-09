#!/bin/bash
#
#SBATCH --job-name=roberta-array_remaining
#SBATCH --array=0-0
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH -t 05:00:00
#SBATCH -p evlab

timestamp() {
  date +"%T"
}

echo "Calculating BLiMP scores for roberta-base-unadjusted"
timestamp

echo "SOURCING minicons-env ENVIRONMENT (ANACONDA39)"
source /om2/user/ckauf/anaconda39/etc/profile.d/conda.sh
conda activate minicons-env

cd /nese/mit/group/evlab/u/ckauf/minicons/carina

export MODELNAME="roberta-large"

i=0
for file in wh_vs_that_no_gap_long_distance ; do #existential_there_object_raising expletive_it_object_raising complex_NP_island ellipsis_n_bar_2 ellipsis_n_bar_1 wh_questions_subject_gap_long_distance sentential_negation_npi_scope ; do
    mkdir -p output/${MODELNAME}/
    file_list[$i]="blimp/data/${file}.jsonl"
    i=$[$i +1]
done

echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "FILE NAME: " ${file_list[$SLURM_ARRAY_TASK_ID]}
echo "MODEL NAME: " $MODELNAME
echo "UNADJUSTED METRIC"

export PYTHONPATH="${PYTHONPATH}:/nese/mit/group/evlab/u/ckauf/minicons/"
python run_BLiMP_exp_individually.py --model $MODELNAME --file ${file_list[$SLURM_ARRAY_TASK_ID]} --batch_size 500

timestamp
echo 'All complete!'
