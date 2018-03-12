# ---- job script ---- #

# activate python environment
export CUDA_VISIBLE_DEVICES=1
source activate pytorch_gpu

# Move to src directory
cd ..

# run main script
# The message variable is not used for now (will be used to update spreadsheet later)
# Variable order: message im_size architecture epochs

python main.py -img_size 224 -b 16 --arch resnet18 -epochs 50 -lr_patience 5 -early_stop 10 -lr 0.0001 --pretrained
