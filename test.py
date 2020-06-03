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
    print('Loading list of test images...')
    test_image_paths = list(np.loadtxt(config['test_ims_file'], dtype='str'))

    # Parameters for deformable registration (with affine initialization)
    print('Loading Elastix registration parameters...')
    parameter_map = [sitk.ReadParameterFile(
        os.path.join(config['param_dir'], f)) for f in sorted(
            os.listdir(config['param_dir']))]

    # Multi-Atlas Segmentation object
    print('Building Multi-Atlas Segmentation model...')
    mas = MultiAtlasSegmentation(atlas_paths, config['atlas_size'],
                                 config['image_metric'],
                                 config['label_fusion'])

    # RCA Classifier object
    print('Building RCA Classifier...')
    rca = SingleAtlasClassifier()

    print('Loop over test images...')
    dice_list = []
    for i, image_path in enumerate(test_image_paths):

        print('{}/{} Image filename: {}'.format(
            i + 1, len(test_image_paths), os.path.basename(image_path)))

        # Predict segmentation
        print('  - Predicting segmentation...')
        label, label_path, idxs = mas.predict_segmentation(
            image_path, out_dir, parameter_map)

        # Predict accuracy
        print('  - Predicting accuracy...')
        rca.image_path = image_path
        rca.label_path = label_path
        dice = rca.predict_dice({
                'images': [atlas_paths['images'][i] for i in idxs],
                'labels': [atlas_paths['labels'][i] for i in idxs]},
                out_dir, parameter_map)
        dice_list.append(dice)

    # Save results
    print('Saving results...')
    with open(os.path.join(
            config['result_dir'], 'rca_predictions.csv'), 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['file', 'dice'])
        writer.writeheader()
        for i in range(len(test_image_paths)):
            writer.writerow({
                'file': os.path.basename(test_image_paths[i]),
                'dice': dice_list[i]
            })


if __name__ == "__main__":
    from utils import read_config_file
    config = read_config_file('./config/JSRT/MAS.cfg')
    test_mas_rca(config)
