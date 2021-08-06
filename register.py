import SimpleITK as sitk
import os

import utils


def register(fixed_image_path, moving_image_path, moving_label_path,
             parameter_map, elastix_out_dir):
    """Image/Label Registration using SimpleElastix."""
    fixed_image = utils.load_image_itk(fixed_image_path)

    moving_images = [utils.load_image_itk(moving_image_path)]
    moving_labels = [utils.load_image_itk(moving_label_path)]

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


def register_landmark(fixed_image_path, moving_image_path,
                      fixed_landmark_path, parameter_map, elastix_out_dir):
    """Image/Landmark Registration using SimpleElastix."""

    fixed_image = utils.load_image_itk(fixed_image_path)
    moving_image = utils.load_image_itk(moving_image_path)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetLogToConsole(False)
    elastixImageFilter.SetLogToFile(False)
    elastixImageFilter.SetOutputDirectory(elastix_out_dir)
    elastixImageFilter.SetFixedImage(fixed_image)
    elastixImageFilter.SetMovingImage(moving_image)

    parameterMapVector = sitk.VectorOfParameterMap()
    parameterMap = sitk.GetDefaultParameterMap('affine')
    parameterMap['FinalBSplineInterpolationOrder'] = ['0']
    parameterMapVector.append(parameterMap)
    parameterMap = sitk.GetDefaultParameterMap('nonrigid')
    parameterMap['FinalBSplineInterpolationOrder'] = ['0']
    parameterMapVector.append(parameterMap)
    elastixImageFilter.SetParameterMap(parameterMapVector)

    elastixImageFilter.Execute()

    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetLogToConsole(False)
    transformixImageFilter.SetLogToFile(False)
    transformixImageFilter.SetOutputDirectory(elastix_out_dir)
    transformParameterMap = elastixImageFilter.GetTransformParameterMap()
    transformixImageFilter.SetTransformParameterMap(transformParameterMap)
    transformixImageFilter.SetFixedPointSetFileName(fixed_landmark_path)

    transformixImageFilter.Execute()

    landmark_file_path = os.path.join(elastix_out_dir, 'outputpoints.txt')
    moving_landmarks = utils.read_landmark_file_elastix(landmark_file_path)

    return moving_landmarks
