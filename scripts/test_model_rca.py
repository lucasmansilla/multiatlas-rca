import os
import argparse
import shutil
import time
import SimpleITK as sitk

from src.utils.io import read_dir, write_csv
from src.models import MAS, SAC


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_images_dir', type=str)
    parser.add_argument('--train_labels_dir', type=str)
    parser.add_argument('--test_images_dir', type=str)
    parser.add_argument('--test_labels_dir', type=str)
    parser.add_argument('--num_atlas', type=int)
    parser.add_argument('--image_measure', type=str)
    parser.add_argument('--label_fusion', type=str)
    parser.add_argument('--eval_metric', type=str)
    parser.add_argument('--results_dir', type=str)
    parser.add_argument('--remove_tmp_dir', action='store_true')
    args = parser.parse_args()

    print('\nArgs:\n')
    for k, v in sorted(vars(args).items()):
        print(f'\t{k}: {v}')


    tmp_dir = os.path.join(args.results_dir, 'tmp')
    os.makedirs(tmp_dir, exist_ok=True)

    # Read list of atlases (i.e., image and its corresponding label)
    atlas_files = {
        'images': read_dir(args.train_images_dir),
        'labels': read_dir(args.train_labels_dir)
    }

    # Read list of input images and their predicted labels
    test_images_list = read_dir(args.test_images_dir)
    test_labels_list = read_dir(args.test_labels_dir)

    # Configure registration procedure
    elx_params = sitk.VectorOfParameterMap()

    affine_params = sitk.GetDefaultParameterMap('affine', 4)
    affine_params['MaximumNumberOfIterations'] = ['500']
    elx_params.append(affine_params)

    bspline_params = sitk.GetDefaultParameterMap('bspline', 4)
    bspline_params['Metric1Weight'] = ['10']
    bspline_params['MaximumNumberOfIterations'] = ['2000']
    elx_params.append(bspline_params)

    # Configure multi-atlas and RCA classifier
    multiatlas = MAS(atlas_files, args.num_atlas, args.image_measure, args.label_fusion)
    classifier = SAC(eval_metric=args.eval_metric)

    results = []

    num_images = len(test_images_list)
    print('\nPredicting accuracy for input segmentations:\n')
    for i, (image_path, label_path) in enumerate(zip(test_images_list, test_labels_list)):
        image_name = os.path.basename(image_path)

        print(f'\t{i+1:>3}/{num_images} File {image_name}', end=' ', flush=True)

        cur_tmp_dir = os.path.join(tmp_dir, f'file_{i+1:0>3}')
        os.makedirs(cur_tmp_dir, exist_ok=True)

        # Select multi-atlas
        t_start = time.time()
        indices = multiatlas.select_atlas(image_path)
        selected_atlas = {
            'images': [atlas_files['images'][i] for i in indices],
            'labels': [atlas_files['labels'][i] for i in indices]
        }

        # Run reverse classifier
        classifier.set_atlas(image_path, label_path)
        pred_score = classifier.predict_accuracy(selected_atlas, elx_params, cur_tmp_dir)
        t_elapsed = time.time() - t_start

        results.append([image_name, pred_score])

        print(f'accuracy: {pred_score:.4f} ({t_elapsed:.2f} sec)')

    if args.remove_tmp_dir:
        shutil.rmtree(tmp_dir)

    # Save results
    header = ['filename', args.eval_metric]
    output_path = os.path.join(args.results_dir, 'result_scores.csv')
    write_csv(results, header, output_path)

    with open(os.path.join(args.results_dir, 'run_args.txt'), 'w') as f:
        for k, v in sorted(vars(args).items()):
            f.write(f'{k}: {v}\n')

    print('\nDone.\n')
