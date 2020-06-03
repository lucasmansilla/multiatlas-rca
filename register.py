import SimpleITK as sitk
import os


def register(fixed_image_path, moving_image_path, moving_label_path,
             parameter_map, elastix_out_dir):
    """Image/Label Registration using SimpleElastix."""
    fixed_image = sitk.ReadImage(fixed_image_path)

    moving_images = [sitk.ReadImage(moving_image_path)]
    moving_labels = [sitk.ReadImage(moving_label_path)]

    for params in parameter_map:
        # Warp moving image
        moving_images.append(
            sitk.Cast(sitk.Elastix(fixed_image, moving_images[-1], params,
                                   False, False, elastix_out_dir),
                      sitk.sitkUInt8))

        transform_params = sitk.ReadParameterFile(
            os.path.join(elastix_out_dir, 'TransformParameters.0.txt'))
        transform_params['FinalBSplineInterpolationOrder'] = ['0']

        # Warp moving label
        moving_labels.append(
            sitk.Cast(sitk.Transformix(moving_labels[-1], transform_params),
                      sitk.sitkUInt8))

    return moving_images[-1], moving_labels[-1]
