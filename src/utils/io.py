import os
import csv
import numpy as np
import SimpleITK as sitk


def read_dir(dir_path):
    """ Read list of files in directory sorted by name. """
    files = [os.path.join(dir_path, f) for f in os.listdir(dir_path)]
    sort_files = sorted(files)
    return sort_files


def read_txt(file_path):
    """ Read list of files in txt file. """
    with open(file_path, 'r') as f:
        files = [line.strip() for line in f]
        return files


def write_csv(data, header, file_path):
    """ Write data into csv file. """
    with open(file_path, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)


def read_points_to_numpy(file_path):
    """ Read SimpleElastix point set file into numpy array. """
    input_points = np.genfromtxt(file_path, skip_header=2)
    result_points = input_points.flatten()
    return result_points


def read_image_to_numpy(file_path):
    """ Read image file into numpy array. """
    image = read_image_to_itk(file_path)
    image = sitk.GetArrayFromImage(image)
    return image


def read_image_to_itk(file_path):
    """ Read image file into ITK image. """
    image = sitk.ReadImage(file_path, sitk.sitkFloat32)
    return image


def write_image(image, file_path, rescale=False):
    """ Write ITK image into file. """
    output_image = sitk.Image(image)
    if rescale:
        output_image = sitk.RescaleIntensity(output_image, 0, 255)
    output_image = sitk.Cast(output_image, sitk.sitkUInt8)
    sitk.WriteImage(output_image, file_path)
