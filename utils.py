import os
import numpy as np


def read_config_file(filename):
    config = {}
    exec(open(filename).read(), config)
    del config['__builtins__']
    return config


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def read_landmark_file_elastix(path):
    data = np.loadtxt(path, dtype=np.str)
    landmarks = []
    for line in list(data):
        landmarks += [float(item) for item in line[27:29]]
    return np.asarray(landmarks)
