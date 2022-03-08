import os
import inspect
import numpy as np
import skimage.transform as ski
import SimpleITK as sitk

import src.fusion
import src.metric
from src.elastix import register_label
from src.utils.io import read_image_to_numpy, read_image_to_itk


class MAS(object):
    """ Multi-atlas model for segmenting images. """

    def __init__(self, atlas_files, num_atlas=5, image_measure='mutual_info', label_fusion='voting'):

        # Check inputs
        image_functions = dict(inspect.getmembers(src.metric.image, inspect.isfunction))
        if image_measure not in image_functions:
            raise ValueError('Invalid similarity measure')

        fusion_functions = dict(inspect.getmembers(src.fusion, inspect.isfunction))
        if label_fusion not in fusion_functions:
            raise ValueError('Invalid fusion method')

        self.atlas_files = atlas_files
        self.num_atlas = num_atlas
        self.image_measure = image_functions[image_measure]
        self.is_similarity = src.metric.image.is_similarity[image_measure]
        self.label_fusion = fusion_functions[label_fusion]

    def predict_label(self, input_path, elx_params, output_dir):
        """ Predict the segmentation of an input image using multi-atlas. """

        # Step 1: Select the multi-atlas
        indices = self.select_atlas(input_path)
        images = [self.atlas_files['images'][i] for i in indices]
        labels = [self.atlas_files['labels'][i] for i in indices]

        # Step 2: Transfer labels via non-rigid registration
        candidate_labels = []
        for i, (image_path, label_path) in enumerate(zip(images, labels)):
            cur_out_dir = os.path.join(output_dir, f'atlas_{i+1:0>3}')
            os.makedirs(cur_out_dir, exist_ok=True)

            output_label = register_label(input_path, image_path, label_path, elx_params, cur_out_dir)[1]
            candidate_labels.append(sitk.Cast(output_label, sitk.sitkUInt8))

        # Step 3: Get result label by fusion
        result_label = self.label_fusion(candidate_labels)

        return result_label, indices

    def select_atlas(self, input_path):
        """ Select a set of atlases by image similarity. """

        input_image = read_image_to_numpy(input_path)

        all_scores = []
        for image_path in self.atlas_files['images']:
            atlas_image = read_image_to_numpy(image_path)

            if input_image.shape != atlas_image.shape:  # resize to atlas image size
                atlas_image = ski.resize(atlas_image, input_image.shape, anti_aliasing=True)

            score = self.image_measure(input_image, atlas_image)
            all_scores.append(score)

        if self.is_similarity:
            # higher for more similar images
            sort_indices = np.argsort(all_scores)[::-1]  # from max value to min value
        else:
            # lower for more similar images
            sort_indices = np.argsort(all_scores)  # from min value to max value

        return_indices = sort_indices[:self.num_atlas]

        return return_indices


class SAC(object):
    """ Single-atlas classifier for evaluating segmentations. """

    def __init__(self, image_path=None, label_path=None, eval_metric='dice_score'):

        # Check inputs
        label_functions = dict(inspect.getmembers(src.metric.label, inspect.isfunction))
        if eval_metric not in label_functions:
            raise ValueError('Invalid evaluation metric')

        self.image_path = image_path
        self.label_path = label_path
        self.eval_metric = label_functions[eval_metric]
        self.is_overlap = src.metric.label.is_overlap[eval_metric]

    def set_atlas(self, image_path, label_path):
        self.image_path = image_path
        self.label_path = label_path

    def predict_accuracy(self, atlas_files, elx_params, output_dir):
        """ Predict the accuracy of an input segmentation. """

        images = atlas_files['images']
        labels = atlas_files['labels']

        # Transfer labels via non-rigid registration and compute metric scores
        candidate_scores = []
        for i, (image_path, label_path) in enumerate(zip(images, labels)):
            cur_out_dir = os.path.join(output_dir, f'atlas_{i+1:0>3}')
            os.makedirs(cur_out_dir, exist_ok=True)

            pred_label = register_label(image_path, self.image_path, self.label_path, elx_params, cur_out_dir)[1]

            true_label = read_image_to_itk(label_path)
            score = self.eval_metric(true_label, pred_label)
            candidate_scores.append(score)

        # Get result score by taking the best value
        if self.is_overlap:
            # highest for overlap metrics
            result_score = np.max(candidate_scores)
        else:
            # lowest for distance metrics
            result_score = np.min(candidate_scores)

        return result_score
