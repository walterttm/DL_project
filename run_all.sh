#!/usr/bin/env bash
# 一键跑 4 组实验
set -e

EPOCHS=${EPOCHS:-30}
BS=${BS:-64}

for exp in A1 A2 B1 B2; do
    echo "================ running $exp ================"
    python train.py --exp $exp --epochs $EPOCHS --batch_size $BS
done

echo "================ plotting ================"
python plot.py
