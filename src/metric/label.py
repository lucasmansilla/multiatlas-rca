import SimpleITK as sitk
import numpy as np

is_overlap = {
    'dice_score': True
}


def dice_score(true_label, pred_label):
    """ Compute the Sørensen-Dice coefficient (DSC) between two segmentations. """

    # Get label classes corresponding to anatomical structures
    classes = np.unique(sitk.GetArrayFromImage(pred_label))
    classes = classes[1:].astype(np.double)  # remove background class (0)

    lom = sitk.LabelOverlapMeasuresImageFilter()

    # Compute Dice score for each anatomical structure
    dice_scores = []
    for i in classes:
        pred_mask = sitk.BinaryThreshold(pred_label, lowerThreshold=i, upperThreshold=i)
        true_mask = sitk.BinaryThreshold(true_label, lowerThreshold=i, upperThreshold=i)

        lom.Execute(pred_mask, true_mask)
        dice_scores.append(lom.GetDiceCoefficient())

    result_dice = np.mean(dice_scores)

    return result_dice
