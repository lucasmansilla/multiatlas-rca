import os


def read_config_file(filename):
    config = {}
    exec(open(filename).read(), config)
    del config['__builtins__']
    return config


def create_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
