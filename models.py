import SimpleITK as sitk
import os
import numpy as np

import utils
from register import register
from binary_metrics import dsc


class MultiAtlasSegmentation(object):
    """Multi-Atlas Segmentation (MAS) model for segmenting chest X-ray
    images."""

    def __init__(self, atlas_paths, atlas_size, image_metric, label_fusion):

        # Check inputs
        import inspect
        import image_metrics
        sim_funcs = dict(
            inspect.getmembers(image_metrics, inspect.isfunction))
        if image_metric not in list(sim_funcs.keys()):
            raise ValueError('Invalid image similarity metric.')

        import fusion_methods
        fus_funcs = dict(
            inspect.getmembers(fusion_methods, inspect.isfunction))
        if label_fusion not in list(fus_funcs.keys()):
            raise ValueError('Invalid label fusion method.')

        self.atlas_paths = atlas_paths
        self.atlas_size = atlas_size
        self.image_metric_func = sim_funcs[image_metric]
        self.is_similarity_metric = image_metrics.is_sim_metric[image_metric]
        self.label_fusion_func = fus_funcs[label_fusion]

    def predict_segmentation(self, image_path, out_dir, parameter_map,
                             remove_tmp_dir=True):
        """Predict the segmentation of a given image using an atlas set."""

        # Temporary directory for Elastix results
        tmp_dir = os.path.join(out_dir, 'tmp')
        utils.create_dir(tmp_dir)

        # Step 1: Atlas selection
        idxs = self._atlas_selection(image_path)

        # Step 2: Registration
        label_list = []
        for atlas_image, atlas_label in zip(
                [self.atlas_paths['images'][i] for i in idxs],
                [self.atlas_paths['labels'][i] for i in idxs]):
            label_list.append(
                register(image_path, atlas_image, atlas_label,
                         parameter_map, tmp_dir)[1])

        # Step 3: Label propagation
        label = self.label_fusion_func(label_list)

        # Save predicted label
        label_path = os.path.join(
            out_dir, os.path.splitext(
                os.path.basename(image_path))[0] + '_mask.png')
        sitk.WriteImage(label, label_path)

        if remove_tmp_dir:
            os.system('rm -rf {0}'.format(tmp_dir))

        return label, label_path, idxs

    def _atlas_selection(self, image_path):
        """Select an atlas set using an image similarity (or dissimilarity)
        measure."""
        image = utils.load_image_arr(image_path)

        scores = []
        for atlas_image in self.atlas_paths['images']:
            scores.append(self.image_metric_func(
                image, utils.load_image_arr(atlas_image)))

        if self.is_similarity_metric:
            # Similarity is higher for more similar images
            return np.argsort(scores)[-self.atlas_size:]
        else:
            # Dissimilarity is lower for more similar images
            return np.argsort(scores)[:self.atlas_size]


class SingleAtlasClassifier(object):
    """Single-Atlas Classifier for predicting accuracy of segmented images
    using the concept of Reverse Classification Accuracy (RCA)."""

    def _init__(self, image_path=None, label_path=None):
        self.image_path = image_path
        self.label_path = label_path

    def predict_dice(self, atlas_paths, out_dir, parameter_map,
                     remove_tmp_dir=True):
        """Predict the Dice score of a given segmentation."""

        # Temporary directory for Elastix results
        tmp_dir = os.path.join(out_dir, 'tmp')
        utils.create_dir(tmp_dir)

        scores = []
        for atlas_image, atlas_label in zip(atlas_paths['images'],
                                            atlas_paths['labels']):
            predicted_label = register(
                atlas_image, self.image_path, self.label_path,
                parameter_map, tmp_dir)[1]
            scores.append(dsc(utils.load_image_itk(atlas_label),
                              predicted_label, True))

        if remove_tmp_dir:
            os.system('rm -rf {0}'.format(tmp_dir))

        return np.max(scores)
