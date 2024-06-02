This repository contains the code used in the Article: IQAEvolNet: A Novel Unsupervised Evolutionary Image Enhancement Algorithm on Chest X-Ray Scans.

The IQAEvolNet code is written on Imports/IQAEvolNet.py file. The application is showcased in the notebooks contained in the repository.

There are 3 notebooks to be accessed:
  - Dehazenet_Enhancement.ipynb: Contains the code for running the IQAEvolNet to train a model for image enhancement. In the first stage of the notebook, the model is pre-trained using the reference algorithms: UM, HEF, CLAHE, ATACE, and TCDHE. The implementation of each of this algorithms can be found on Imports/enhancement_algorithms.py. The resulting pre-trained weights are saved on Datasets/TestCXR/Pretrained_weights/lightdehazeTestnet/name_of_the_algorithm, with name_of_the_algorithm being: um, hef, clahe, atace, tcdhe. The second stage of the notebook runs the evolutionary algorithm. The results are stored on Enhancement/TestCXR/complete/weight_set_id.
  - ClassificationReferenceMethods.ipynb: Contains the code for running the classification algorithm using each of the reference algorithms as pre-processing. The results are saved on IQAEvolNet/Classification/TestCXR/complete/name_of_the_algorithm.
  - ClassificationLdnet.ipynb: Contains the code for running the classification algorithm using one of the models trained by IQAEvolNet. The results are saved on IQAEvolNet/Classification/TestCXR/complete/weight_set_id.
