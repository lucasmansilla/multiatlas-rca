import SimpleITK as sitk
import numpy as np


def dsc(result, reference, average=False):
    """Sørensen-Dice Coefficient (DSC) between two masks with SimpleElastix."""

    # Get the labels of the anatomical structures
    struct_labels = np.unique(
        sitk.GetArrayFromImage(result))[1:].astype(np.double)

    lom_filter = sitk.LabelOverlapMeasuresImageFilter()

    # Compute the value of DSC for each anatomical structure
    scores = []
    for label in struct_labels:
        result_bin = sitk.BinaryThreshold(
            result, lowerThreshold=label, upperThreshold=label,
            insideValue=1, outsideValue=0)
        reference_bin = sitk.BinaryThreshold(
            reference, lowerThreshold=label, upperThreshold=label,
            insideValue=1, outsideValue=0)

        lom_filter.Execute(result_bin, reference_bin)
        scores.append(lom_filter.GetDiceCoefficient())

    return scores if not average else np.mean(scores)
