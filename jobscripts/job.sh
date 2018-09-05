# ---- job script ---- #

# activate python environment
export CUDA_VISIBLE_DEVICES=2
source activate otoliths-identification-env

# Move to src directory
cd ..

# run main script
# The message variable is not used for now (will be used to update spreadsheet later)
# Variable order: message im_size architecture epochs

python main_hier.py --train --test
