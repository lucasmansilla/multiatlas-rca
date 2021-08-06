def test_mas_rca(config):

    import numpy as np
    import os
    import SimpleITK as sitk
    import csv
    from models import MultiAtlasSegmentation, SingleAtlasClassifier
    from utils import create_dir

    create_dir(config['result_dir'])
    out_dir = os.path.join(config['result_dir'], 'labels')  # output labels
    create_dir(out_dir)

    # Training images and labels
    atlas_paths = {
        'images': list(np.loadtxt(config['train_ims_file'], dtype='str')),
        'labels': list(np.loadtxt(config['train_lbs_file'], dtype='str'))
    }

    # Test images (to be segmented)
    print('Loading list of test images', end=' ', flush=True)
    test_image_paths = list(np.loadtxt(config['test_ims_file'], dtype='str'))
    print('Ok')

    # Parameters for deformable registration (with affine initialization)
    print('Loading Elastix registration parameters', end=' ', flush=True)
    parameter_map = [sitk.ReadParameterFile(
        os.path.join(config['param_dir'], f)) for f in sorted(
            os.listdir(config['param_dir']))]
    print('Ok')

    # Multi-Atlas Segmentation object
    print('Building Multi-Atlas Segmentation model', end=' ', flush=True)
    mas = MultiAtlasSegmentation(atlas_paths, config['atlas_size'],
                                 config['image_metric'],
                                 config['label_fusion'])
    print('Ok')

    # RCA Classifier object
    print('Building RCA Classifier', end=' ', flush=True)
    rca = SingleAtlasClassifier()
    print('Ok')

    print('Processing test images')
    dice_list = []
    for i, image_path in enumerate(test_image_paths):

        print('{}/{} Image filename: {}'.format(
            i + 1, len(test_image_paths), os.path.basename(image_path)))

        # Predict segmentation
        print('  - Predicting segmentation', end=' ', flush=True)
        label, label_path, idxs = mas.predict_segmentation(
            image_path, out_dir, parameter_map)
        print('Ok')

        # Predict accuracy
        print('  - Predicting accuracy', end=' ', flush=True)
        rca.image_path = image_path
        rca.label_path = label_path
        dice = rca.predict_dice({
            'images': [atlas_paths['images'][i] for i in idxs],
            'labels': [atlas_paths['labels'][i] for i in idxs]},
            out_dir, parameter_map)
        dice_list.append(dice)
        print('Ok')

    # Save results
    print('Saving results', end=' ', flush=True)
    with open(os.path.join(
            config['result_dir'], 'rca_predictions.csv'), 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'dice'])
        writer.writeheader()
        for i in range(len(test_image_paths)):
            writer.writerow({
                'file': os.path.basename(test_image_paths[i]),
                'dice': dice_list[i]
            })
    print('Ok')


if __name__ == "__main__":
    from utils import read_config_file
    config = read_config_file('./config/JSRT/MAS.cfg')
    test_mas_rca(config)
