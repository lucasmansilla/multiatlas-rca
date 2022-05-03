# Image segmentation and accuracy prediction via Multi-atlas segmentation (MAS) and Reverse classification accuracy (RCA)

This repository contains an extended version of the source code corresponding to the paper "Segmentación multi-atlas de imágenes médicas con selección de atlas inteligente y control de calidad automático" (La Plata, 2018). You can check out or paper here: http://sedici.unlp.edu.ar/handle/10915/73180.

## Description
The most salient aspects of the project are as follows:
- Atlas selection by image similarity. Available image measures are: Mean absolute error (MAE), Mean squared error (MSE), Normalized cross correlation (NCC) and Mutual information (MI). 
- Deformable image registration (with affine initialization) via [SimpleElastix](https://simpleelastix.github.io/).
- Two label fusion techniques: Voting and STAPLE.
- Quality evaluation for predicted segmentations via [RCA](https://arxiv.org/abs/1702.03407).

## Instructions
This project uses Python 3.8.10.

### Project environment:
1. Create and activate virtual environment: 1) `python3 -m venv env` 2) `source env/bin/activate`
2. Install required packages: `pip install -r requirements.txt`
3. Install project modules (src): `pip install -e .`
4. Install SimpleElastix toolbox following [this guide](https://gist.github.com/vfmatzkin/0fcc79a61f9bafcc2113fd83a8900937).

### Simulations:
- Multi-atlas: `./01_run_multiatlas.sh`
- RCA: `./02_run_rca.sh`

## Reference
- Mansilla, L., & Ferrante, E. (2018). Segmentación multi-atlas de imágenes médicas con selección de atlas inteligente y control de calidad automático. In XXIV Congreso Argentino de Ciencias de la Computación (La Plata, 2018).

## License
[MIT](https://choosealicense.com/licenses/mit/)
