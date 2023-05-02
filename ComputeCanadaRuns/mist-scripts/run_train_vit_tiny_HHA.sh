#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=7:58:58
#SBATCH --job-name vit_tiny_HHA
#SBATCH --mail-user=zeinhajjali@sce.carleton.ca
#SBATCH --mail-type=ALL
#SBATCH --account=soscip-3-091
# ----------------------------------------------------------

# Make the required folders in the temporary directory
mkdir -p $SLURM_TMPDIR/data/current/train/noone $SLURM_TMPDIR/data/current/train/nurse $SLURM_TMPDIR/data/current/val/noone $SLURM_TMPDIR/data/current/val/nurse $SLURM_TMPDIR/savedModels

# Copy the torch pretrained network from the scratch folder to the temp directory
cp -r $SLURM_SUBMIT_DIR/torchHome $SLURM_TMPDIR

# Copy and unzip the data
cp $SLURM_SUBMIT_DIR/data/HHA_Depth_prePT.zip $SLURM_TMPDIR/data
cd $SLURM_TMPDIR/data
unzip HHA_Depth_prePT.zip

# Copy the training script
cp $SLURM_SUBMIT_DIR/train_vit_tiny_HHA.py $SLURM_TMPDIR

# Set the starting directory
cd $SLURM_TMPDIR

module load anaconda3

source activate dCNN

# Run the training script with run number=1
python train_vit_tiny_HHA.py $NUM 1

# Copy the output model back to the main slurm folder
cp $SLURM_TMPDIR/savedModels/* $SLURM_SUBMIT_DIR/savedModels/vit_tiny/HHA

num=$NUM
if [ "$num" -lt 5 ]; then
      num=$(($num+1))
      ssh -t mist-login01 "cd $SLURM_SUBMIT_DIR; sbatch --export=NUM=$num run_train_vit_tiny_HHA.sh";
fi
