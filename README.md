# AGender
Age and Gender Detection using Python

### Description
Age and Gender detection using OpenCV, Keras, Tensorflow and Wiki-crop dataset and imbd dataset. 
OpenCV is used to detect face,crop and to grayscale image. Haar-cascade frontal face classifier is used to detect face for complex image.\
A custom Convolutional Neural Network is used in this purpose. Any CNN will work. The CNN implementation is on **CNN.py**. 
Dataset preprocessing and cleaning is done on **Dataset.py**.\
**agender.py** is the main pyhton script. It uses the pretrained model **agender.h5**.\
Training the model is done using **Jupyter Notebooks** for easy visualisation and presentataion. \

### Datset
https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/
Download the dataset, extract and place it on the **Dataset** folder. Be sure to add the **.mat** file.
