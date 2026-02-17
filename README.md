Overview
--------

This project implements a bird species classification system using deep feature extraction with ResNet50 (transfer learning) combined with traditional machine learning classifiers.

The objective was to classify images into one of 200 bird species using pretrained convolutional neural network features and evaluate multiple classifiers to determine optimal performance.

The best model achieved 86% classification accuracy using Logistic Regression on PCA-reduced feature vectors.

Dataset
-------

This project uses the Caltech-UCSD Birds 200 (CUB-200) dataset.

200 bird species

6033 labeled images

~20–30 images per class

Dataset Source:
http://www.vision.caltech.edu/visipedia/CUB-200.html

Note: The dataset is publicly available for research purposes and is not included in this repository.

Technologies Used
-----------------

Python

TensorFlow

Keras

OpenCV

NumPy

Scikit-learn

PCA (Principal Component Analysis)

Methodology
-----------

1. Image Preprocessing

Images resized to match ResNet50 input dimensions

Preprocessing applied using preprocess_input()

Normalization performed before feature extraction

2. Feature Extraction (Transfer Learning)

Used pretrained ResNet50 (ImageNet weights)

include_top=False to remove final classification layers

Added Global Average Pooling layer

Extracted 2048-dimensional feature vectors

Stored extracted features in features.npy

Stored corresponding labels in labels.npy

3. Dimensionality Reduction

Applied PCA to reduce feature dimensionality

Reduced 2048 → 1024 dimensions

Observed improved classification performance at 1024 components

4. Classification Models Evaluated

Model	Accuracy
Logistic Regression	86%
Neural Network Classifier	78%
K-Nearest Neighbors	72%
Support Vector Classifier	61%
Decision Tree	45%

The highest performance was achieved using Logistic Regression (86% accuracy).

Train-Test Split
----------------

80% Training Data

20% Test Data

Evaluation Metrics
------------------

Accuracy

Mean Squared Error (MSE)

Due to the presence of 200 distinct classes, confusion matrix visualization was limited.

Results
-------

The final model achieved approximately 86% classification accuracy.

The system successfully classified unseen bird images downloaded from external sources.

Key Learnings
-------------

Practical implementation of transfer learning

Feature extraction using pretrained CNNs

Dimensionality reduction using PCA

Performance comparison of multiple classifiers

Hybrid deep learning + classical ML pipeline

Project Context
---------------

Developed as part of the Certified AI Professional program conducted by National Institute of Electronics and Information Technology, Calicut.

This project was independently implemented to apply deep learning and machine learning concepts in a real-world image classification task.
