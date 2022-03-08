#!/bin/bash

python scripts/test_model_rca.py --train_images_dir=data/JSRT/train/images \
                                 --train_labels_dir=data/JSRT/train/labels \
                                 --test_images_dir=data/JSRT/test/images \
                                 --test_labels_dir=results/test_mas/JSRT/output_labels \
                                 --results_dir=results/test_rca/JSRT \
                                 --num_atlas=5 \
                                 --image_measure='mutual_info' \
                                 --label_fusion='voting' \
                                 --eval_metric='dice_score' \
                                 --remove_tmp_dir