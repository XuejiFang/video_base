#!/bin/bash
export SCRIPT=${1:-example.sh}
# 定义总共有 8 个 GPU
for RANK in {0..7}
do
    # 在后台启动 eval.sh 脚本，传递不同的 RANK
    bash  $SCRIPT $RANK &
done

# 等待所有后台进程完成
wait
