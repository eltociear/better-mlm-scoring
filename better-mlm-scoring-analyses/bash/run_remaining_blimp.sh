#!/bin/bash
#
#SBATCH --job-name=bert-array_highmem
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH -t 04:00:00
#SBATCH -p evlab

timestamp() {
  date +"%T"
}

export MODELNAME="bert-base-cased"
export MASKING="original"
timestamp

echo "SOURCING minicons-env ENVIRONMENT (ANACONDA39)"
source /om2/user/ckauf/anaconda39/etc/profile.d/conda.sh
conda activate minicons-env

cd /nese/mit/group/evlab/u/ckauf/minicons/carina

i=0
#for file in principle_A_c_command existential_there_subject_raising wh_vs_that_no_gap_long_distance only_npi_scope existential_there_quantifiers_2 existential_there_quantifiers_1 distractor_agreement_relational_noun distractor_agreement_relative_clause ; do
#for file in wh_vs_that_no_gap_long_distance ; do
for file in wh_vs_that_no_gap wh_vs_that_with_gap_long_distance existential_there_quantifiers_1 ; do
    mkdir -p output/${MODELNAME}/
    file_list[$i]="blimp/data/${file}.jsonl"
    i=$[$i +1]
done

echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "FILE NAME: " ${file_list[$SLURM_ARRAY_TASK_ID]}
echo "MASKING NAME: " $MASKING
echo "MODEL NAME: " $MODELNAME

export PYTHONPATH="${PYTHONPATH}:/nese/mit/group/evlab/u/ckauf/minicons/"
python run_BLiMP_exp_individually.py --model $MODELNAME --file ${file_list[$SLURM_ARRAY_TASK_ID]} --which_masking $MASKING --batch_size 500

timestamp
echo 'All complete!'

