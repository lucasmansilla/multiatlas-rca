import SimpleITK as sitk
import os
import numpy as np

import utils
from elastix import register
from binary_metrics import dsc


class MultiAtlasSegmentation(object):
    """ Multi-Atlas Segmentation (MAS) model for segmenting chest X-ray images. """

    def __init__(self, atlas_paths, atlas_size, image_metric, label_fusion):
        import inspect
        import image_metrics
        import fusion_methods

        # Check inputs
        similarity_funcs = dict(inspect.getmembers(image_metrics, inspect.isfunction))
        if image_metric not in list(similarity_funcs.keys()):
            raise ValueError('Invalid image similarity metric')
        fusion_funcs = dict(inspect.getmembers(fusion_methods, inspect.isfunction))
        if label_fusion not in list(fusion_funcs.keys()):
            raise ValueError('Invalid label fusion method')

        self.atlas_paths = atlas_paths
        self.atlas_size = atlas_size
        self.image_metric_func = similarity_funcs[image_metric]
        self.is_similarity_metric = image_metrics.is_sim_metric[image_metric]
        self.label_fusion_func = fusion_funcs[label_fusion]

    def predict_segmentation(self, image_path, output_dir, parameter_map_lst, remove_tmp_dir=True):
        """ Predict the segmentation for a given image using an atlas set. """

        # Temporary directory for saving Elastix results
        tmp_dir = os.path.join(output_dir, 'tmp')
        utils.create_dir(tmp_dir)

        # Step 1: Atlas selection
        atlas_idxs = self._atlas_selection(image_path)
        images_lst = [self.atlas_paths['images'][i] for i in atlas_idxs]
        labels_lst = [self.atlas_paths['labels'][i] for i in atlas_idxs]

        # Step 2: Registration
        result_labels = []
        for atlas_image, atlas_label in zip(images_lst, labels_lst):
            result_label = register(
                image_path, atlas_image, atlas_label, parameter_map_lst, tmp_dir)[1]
            result_labels.append(sitk.Cast(result_label, sitk.sitkUInt8))

        # Step 3: Label propagation
        predicted_label = self.label_fusion_func(result_labels)

        # Save the predicted label image
        predicted_label_path = os.path.join(
            output_dir, os.path.splitext(os.path.basename(image_path))[0] + '_labels.png')
        sitk.WriteImage(predicted_label, predicted_label_path)

        if remove_tmp_dir:
            os.system('rm -rf {0}'.format(tmp_dir))

        return predicted_label, predicted_label_path, atlas_idxs

    def _atlas_selection(self, image_path):
        """ Select the atlas set using an image similarity (or dissimilarity) measure. """

        image = utils.read_image_arr(image_path)

        scores = []
        for atlas_image in self.atlas_paths['images']:
            scores.append(self.image_metric_func(image, utils.read_image_arr(atlas_image)))

        if self.is_similarity_metric:
            # Similarity is higher for more similar images
            return np.argsort(scores)[-self.atlas_size:]
        else:
            # Dissimilarity is lower for more similar images
            return np.argsort(scores)[:self.atlas_size]


class SingleAtlasClassifier(object):
    """ Single-Atlas Classifier for predicting the accuracy of segmented images using the concept
        of Reverse Classification Accuracy (RCA). """

    def _init__(self, image_path=None, label_path=None):
        self.image_path = image_path
        self.label_path = label_path

    def predict_dice(self, atlas_paths, output_dir, parameter_map_lst, remove_tmp_dir=True):
        """ Predict the Dice score for a given segmentation. """

        # Temporary directory for saving Elastix results
        tmp_dir = os.path.join(output_dir, 'tmp')
        utils.create_dir(tmp_dir)

        scores = []
        for atlas_image, atlas_label in zip(atlas_paths['images'], atlas_paths['labels']):
            predicted_label = register(
                atlas_image, self.image_path, self.label_path, parameter_map_lst, tmp_dir)[1]
            scores.append(dsc(utils.read_image_itk(atlas_label), predicted_label, True))

        if remove_tmp_dir:
            os.system('rm -rf {0}'.format(tmp_dir))

        return np.max(scores)

    def set_atlas_path(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path
