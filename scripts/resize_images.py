import os
import argparse
from PIL import Image

from src.utils.io import read_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str)
    parser.add_argument('--width', type=float, default=256)
    parser.add_argument('--height', type=int, default=256)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    images_files = read_dir(args.images_dir)

    num_images = len(images_files)
    print('\nResizing images:\n')
    for i, image_path in enumerate(images_files):
        image_name = os.path.basename(image_path)

        print(f'\t{i+1:>3}/{num_images} File {image_name}', end=' ', flush=True)

        in_image = Image.open(image_path)

        # Resize image and save
        out_image = in_image.resize((args.width, args.height), Image.BILINEAR)
        out_image.save(os.path.join(args.output_dir, image_name))

        print('Ok')

    print('\nDone.\n')
