import SimpleITK as sitk


def voting(labels):
    """ Majority voting. """
    return sitk.LabelVoting(labels, 0)


def staple(labels):
    """ Simultaneous truth And performance level estimation (STAPLE). """
    return sitk.MultiLabelSTAPLE(labels, 0)
