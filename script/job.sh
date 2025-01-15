#!/bin/bash
#SBATCH --job-name=train_job            # 作业名称
#SBATCH --partition=qilab               # 指定分区
#SBATCH --gres=gpu:8                    # 申请8张GPU
#SBATCH --ntasks=1                      # 单任务作业
#SBATCH --cpus-per-task=128             # 每个任务分配的CPU核心数，根据需求调整
#SBATCH --mem=1000G                     # 分配内存大小
#SBATCH --time=999:00:00                # 设置最大运行时间

export SCRIPT=${1:-"script/train.sh"}
export CONFIG=${2:-"config/example.yaml"}

clear

# # 加载必要模块或环境变量
# module load cuda/12.5  # 根据系统支持的 CUDA 版本调整
# module load nccl/2.22.3-cuda-12.5

# # 激活 Conda 环境
# source activate /storage/qiguojunLab/fangxueji/envs/opensoraplan

# 确保工作目录正确
cd /storage/qiguojunLab/fangxueji/Projects/iccv25

# 打印信息（用于调试）
echo "Job started on $(hostname) at $(date)"

# 启动训练脚本
bash $SCRIPT $CONFIG

# 作业完成提示
echo "Job finished on $(hostname) at $(date)"