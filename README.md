# Image Segmentation and Accuracy Prediction via Multi-Atlas Segmentation (MAS) and Reverse Classification Accuracy (RCA)

This repository contains an extended version of the source code corresponding to the paper "Segmentación multi-atlas de imágenes médicas con selección de atlas inteligente y control de calidad automático" (La Plata, 2018). You can read or download it from [here](http://sedici.unlp.edu.ar/handle/10915/73180).

## Description
The most salient aspects of the project are as follows:
- Atlas selection by image similarity (or dissimilarity). Available metrics are: Sum of Absolute Differences (SAD), Sum of Squared Differences (SSD), Normalized Cross Correlation (NCC) and Mutual Information (MI). 
- Deformable image registration (with affine initialization) with [SimpleElastix](https://simpleelastix.github.io/).
- Two label fusion techniques: Voting and STAPLE.
- Predicting Dice Similarity Coefficient (DSC) values for predicted segmentations with [RCA](https://arxiv.org/abs/1702.03407).

## Requirements
To run the code, you need to install Python 3 and the following libraries:
- [SimpleElastix](https://simpleelastix.github.io/)
- [NumPy](https://numpy.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)

## Reference
- Mansilla, L., & Ferrante, E. (2018). Segmentación multi-atlas de imágenes médicas con selección de atlas inteligente y control de calidad automático. In XXIV Congreso Argentino de Ciencias de la Computación (La Plata, 2018).

## License
[MIT](https://choosealicense.com/licenses/mit/)
