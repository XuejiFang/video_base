clear
export CONFIG=${1:-"config/example.yaml"}

export PATH="/storage/qiguojunLab/qiguojun/anaconda3/bin:$PATH"
conda init
source ~/.bashrc
conda activate /storage/qiguojunLab/fangxueji/envs/opensoraplan

export WANDB_MODE="offline"
accelerate launch \
    --config_file script/configs/deepspeed.yaml \
    train.py \
    --config $CONFIG