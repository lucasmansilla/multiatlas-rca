import SimpleITK as sitk


def voting(pred_labels_list):
    """Label fusion using mayority voting."""
    return sitk.LabelVoting(pred_labels_list, 0)


def staple(pred_labels_list):
    """Label fusion using Simultaneous Truth And Performance
    Level Estimation (STAPLE)."""
    return sitk.MultiLabelSTAPLE(pred_labels_list, 0)
