import numpy as np
import os
import SimpleITK as sitk
import csv
from models import MultiAtlasSegmentation, SingleAtlasClassifier
from utils import create_dir


def main(config):

    create_dir(config['result_dir'])
    output_dir = os.path.join(config['result_dir'], 'labels')
    create_dir(output_dir)

    atlas_paths = {
        'images': list(np.loadtxt(config['train_ims_file'], dtype='str')),
        'labels': list(np.loadtxt(config['train_lbs_file'], dtype='str'))
    }

    print('Loading test images', end=' ', flush=True)
    test_image_paths = list(np.loadtxt(config['test_ims_file'], dtype='str'))
    print('Ok')

    print('Loading Elastix registration parameters', end=' ', flush=True)
    parameter_map_lst = [sitk.ReadParameterFile(os.path.join(config['param_dir'], f))
                         for f in sorted(os.listdir(config['param_dir']))]
    print('Ok')

    print('Building Multi-Atlas Segmentation model', end=' ', flush=True)
    mas = MultiAtlasSegmentation(atlas_paths, config['atlas_size'],
                                 config['image_metric'],
                                 config['label_fusion'])
    print('Ok')

    print('Building RCA Classifier', end=' ', flush=True)
    rca = SingleAtlasClassifier()
    print('Ok')

    print('Processing test images')
    dice_scores = []
    for i, image_path in enumerate(test_image_paths):

        print('{}/{} Image filename: {}'.format(
            i + 1, len(test_image_paths), os.path.basename(image_path)))

        # Predict segmentation
        print('  - Predicting segmentation', end=' ', flush=True)
        label, label_path, atlas_idxs = mas.predict_segmentation(
            image_path, output_dir, parameter_map_lst)
        print('Ok')

        atlas_paths_rca = {
            'images': [atlas_paths['images'][i] for i in atlas_idxs],
            'labels': [atlas_paths['labels'][i] for i in atlas_idxs]
        }

        # Predict accuracy
        print('  - Predicting accuracy', end=' ', flush=True)
        rca.set_atlas_path(image_path, label_path)
        dice = rca.predict_dice(atlas_paths_rca, output_dir, parameter_map_lst)
        dice_scores.append(dice)
        print('Ok')

    # Save results
    print('Saving results', end=' ', flush=True)
    with open(os.path.join(config['result_dir'], 'rca_predictions.csv'), 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'dice'])
        writer.writeheader()
        for i in range(len(test_image_paths)):
            writer.writerow({
                'file': os.path.basename(test_image_paths[i]),
                'dice': dice_scores[i]
            })
    print('Ok')


if __name__ == "__main__":
    from utils import read_config_file
    config = read_config_file('./config/JSRT/MAS.cfg')
    main(config)
