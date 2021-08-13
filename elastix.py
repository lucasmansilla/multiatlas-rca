import SimpleITK as sitk

import utils


def register(fixed_image_path, moving_image_path, moving_label_path,
             parameter_map_lst, output_dir):
    """Image registration and segmentation transformation via SimpleElastix."""

    fixed_image = utils.read_image_itk(fixed_image_path)
    moving_image = utils.read_image_itk(moving_image_path)
    moving_label = utils.read_image_itk(moving_label_path)

    elx_filter = sitk.ElastixImageFilter()
    elx_filter.SetFixedImage(fixed_image)
    elx_filter.SetMovingImage(moving_image)

    # Set the parameters for registering the images
    parameter_map_vector = sitk.VectorOfParameterMap()
    parameter_map_vector = [pm for pm in parameter_map_lst]
    elx_filter.SetParameterMap(parameter_map_vector)

    elx_filter.SetLogToConsole(False)
    elx_filter.SetLogToFile(False)
    elx_filter.SetOutputDirectory(output_dir)

    # Register the moving image to the fixed image
    elx_filter.Execute()

    tfx_filter = sitk.TransformixImageFilter()

    # Set the transform parameters for transforming the moving label image
    transform_parameter_map = elx_filter.GetTransformParameterMap()
    for i in range(len(transform_parameter_map)):
        transform_parameter_map[i]['FinalBSplineInterpolationOrder'] = ['0']
    tfx_filter.SetTransformParameterMap(transform_parameter_map)

    tfx_filter.SetMovingImage(moving_label)

    tfx_filter.SetLogToConsole(False)
    tfx_filter.SetLogToFile(False)
    tfx_filter.SetOutputDirectory(output_dir)

    # Transform the moving label image
    tfx_filter.Execute()

    result_image = elx_filter.GetResultImage()
    result_label = tfx_filter.GetResultImage()

    return result_image, result_label


def register_landmarks(fixed_image_path, moving_image_path,
                       fixed_landmarks_path, output_dir):
    """Image registration and landmark transformation via SimpleElastix."""

    fixed_image = utils.read_image_itk(fixed_image_path)
    moving_image = utils.read_image_itk(moving_image_path)

    elx_filter = sitk.ElastixImageFilter()
    elx_filter.SetFixedImage(fixed_image)
    elx_filter.SetMovingImage(moving_image)

    # Set the parameters for registering the images
    parameter_map_vector = sitk.VectorOfParameterMap()
    parameter_map_vector = [sitk.GetDefaultParameterMap(tf) for tf in ['affine', 'nonrigid']]
    elx_filter.SetParameterMap(parameter_map_vector)

    elx_filter.SetLogToConsole(False)
    elx_filter.SetLogToFile(False)
    elx_filter.SetOutputDirectory(output_dir)

    # Register the moving image to the fixed image
    elx_filter.Execute()

    tfx_filter = sitk.TransformixImageFilter()

    # Set the transform parameters for transforming the fixed image landmarks
    transform_parameter_map = elx_filter.GetTransformParameterMap()
    tfx_filter.SetTransformParameterMap(transform_parameter_map)

    tfx_filter.SetFixedPointSetFileName(fixed_landmarks_path)

    tfx_filter.SetLogToConsole(False)
    tfx_filter.SetLogToFile(False)
    tfx_filter.SetOutputDirectory(output_dir)

    # Transform the fixed image landmarks
    tfx_filter.Execute()

    moving_landmarks = utils.read_landmarks_itk(output_dir + '/outputpoints.txt')

    return moving_landmarks
