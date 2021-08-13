import SimpleITK as sitk


def voting(predicted_labels_lst):
    """ Label fusion using mayority voting. """
    return sitk.LabelVoting(predicted_labels_lst, 0)


def staple(predicted_labels_lst):
    """ Label fusion using Simultaneous Truth And Performance Level Estimation (STAPLE). """
    return sitk.MultiLabelSTAPLE(predicted_labels_lst, 0)
