import SimpleITK as sitk
import numpy as np


def dsc(result, reference, average=False):
    """ Sørensen-Dice Coefficient (DSC) between two segmentations. """

    # Get the labels corresponding to the anatomical structures
    labels = np.unique(sitk.GetArrayFromImage(result))[1:].astype(np.double)

    lom_filter = sitk.LabelOverlapMeasuresImageFilter()

    # Compute the Dice score for each anatomical structure
    scores = []
    for label in labels:
        y_pred = sitk.BinaryThreshold(result, lowerThreshold=label, upperThreshold=label)
        y_true = sitk.BinaryThreshold(reference, lowerThreshold=label, upperThreshold=label)
        lom_filter.Execute(y_pred, y_true)
        scores.append(lom_filter.GetDiceCoefficient())

    return scores if not average else np.mean(scores)
