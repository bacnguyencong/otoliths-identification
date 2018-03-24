# ---- job script ---- #

# activate python environment
export CUDA_VISIBLE_DEVICES=2
source activate pytorch_gpu

# Move to src directory
cd ..

# run main script
# The message variable is not used for now (will be used to update spreadsheet later)
# Variable order: message im_size architecture epochs

python main.py -img_size 224 -b 32 -j 4 --arch densenet121 -epochs 100 -lr_patience 5 -early_stop 10 -lr 0.001
