#!/usr/bin/env bash

python main.py --data <path_to_imagenet_dataset> --model condensenet_cell -j 10 -b 128 --epochs 120 --stages 4-6-8-10-8
    --growth 8-16-32-64-128 --bottleneck 4 --group-1x1 4 --group-3x3 8 --condense-factor 4 --group-lasso-lambda 0.00001
    --cell_stages 0-0-6-7-6 --init_div_groups --cell_type TreeCellB --gpu 0,1 --bn_before_add --use_avg
    --evaluate-from <path_to_nets/ImageNet#CondenseTreeCellB.pth.tar>

