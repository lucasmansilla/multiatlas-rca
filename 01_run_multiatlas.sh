#!/bin/bash

python scripts/test_model_mas.py --train_images_dir=data/JSRT/train/images \
                                 --train_labels_dir=data/JSRT/train/labels \
                                 --test_images_dir=data/JSRT/test/images \
                                 --results_dir=results/test_mas/JSRT \
                                 --num_atlas=5 \
                                 --image_measure='mutual_info' \
                                 --label_fusion='voting' \
                                 --remove_tmp_dir