# SEMANTIC SEGMENTATION OF AERIAL FARMLAND IMAGES USING DEEP LEARNING ARCHITECTURE

## Abstract 

The application of technology in farming has expanded dramatically over the past few years. Here aerial images are captured by the agricultural drones and these
pictures are fed to advanced convolutional neural networks to analyse the agricultural patterns. As a pilot study of aerial agricultural pattern classification,
various deep learning-based segmentation models are studied and implemented. This project selected
three popular models to fit agricultural data and results are evaluated.This technology will enable us to take full advantage of this technology in monitoring the real-time farmlands and identifying patterns to quickly respond and take action. Doing this can reduce the usage of herbicides in farmlands
thus reducing labour costs and environmental impacts.



Reasearch paper is available [here](https://github.com/HishamParol/DeepLearning-AerialFarmLand/blob/master/ResearchPaper.pdf)

## How to setup and run the project

Semantic segmentation model of agricultural farmland images are build using Python's Django framework. This function will run on multiple Operating Systems including Windows, Linux, and MacOS.

See installation for full installation instructions. You will need a modern GPU with CUDA support for best performance. AMD GPUs are partially supported.

This project has two entry points:
1. Train a model from scratch and see the evaluation metrics. 

2. Predict a new test image and see the results. 

## Resources

In order to complete this project, the following resources were used:

### Integrated Development Environment (IDE):
1. Google Colab Pro - A specialized version of the
Jupyter Notebook provided by Google. This runs on the Cloud Platform and provides priority access to
faster GPUs such as T4 and P100 GPUs. Also, Colab Pro provides access to high memory VMs. This
helped to train the neural network without exhausting the resources.

2. Django Framework

### Data Storage:
Google Drive - Drive provides 30 GB of free memory without a subscription. Also, Drive can be
easily mounted to Colab making it easier to access data quickly.

### Framework:
This project is built using the TensorFlow framework - an open-source library developed
by Google specifically used for training and inference of deep learning networks.
1. Keras: Keras is used as an interface for TensorFlow. This API developed by Google is used
for implementing neural networks
2. TensorBoard: This is used for visualization and tracking the evaluation metrics such as loss
functions and train and validation datasets.

## Data

The data needed for this project was provided by [Agricultural-Vision](https://www.agriculture-vision.com/agriculture-vision-2020/dataset)  -an independent research
board for the research and development of computer vision technology for agriculture. The dataset is
large scale and high quality images of aerial farmlands for the advanced study of agricultural patterns.
The dataset was precisely annotated by professional agronomists with a strict quality assurance process.
The proposed agricultural vision contains 94,986 samples of images and the subset of this dataset
containing 12,901 samples was used for challenge competition. Same was used for this project as well.
The dataset contains nine types of annotations - double plant, dry down, endow, nutrient deficiency,
planter skip, storm damage, water, waterway and weed cluster. In this project only weed clusters are
used for the segmentation task. These images were captured by special mounted camera on Drones
flown over various corn and soybean fields around Illinois and Iowa (USA).
## Building a semantic segmentation deep learning model using TensorFlow

1. Data Pre-processing
.1. Data statistics
.2. Choosing images with weed clusters
.3. Data Augmentation
..1. Geometric Augmentation
..2. Color Augmentation
2. Build and Train models
.1. U-Net Architecture
.2. FCN Architecture\
.3. DeepLabv3+ Architecture
4. Transfer Learning
5. Evaluation results
6. Prediction 

## Model Architecture

### Concept of Encoder-Decoder Network

The most important concept used in semantic segmentation is explained in detail. In recent years, the convolutional neural network (CNN) has made remarkable achievements in semantic segmentation. Nowadays most semantic segmentation networks are based on the concept of encoder-decoder architecture, where it has an encoder side to extract the feature vectors and a decoder side for recovering feature map resolution.

In regular Image Classification Deep Convolutional Neural Network (DCNN) models, it takes images as input and outputs a single value representing the label of that image. It has four main operations, namely - Convolutions, activation function, pooling, and fully connected layers. When we pass the image through these four layers it presents a feature vector with probabilities of each class. In this network, we assign a single label to an entire image. In classification problems, we don't care much
about spatial location. Only the presence of a class label is determined. But in segmentation, it is very important to preserve spatial information. Here we want to categorize each pixel in that image. Understanding images at the pixel level is important here. Hence regular DCNN Models are not suitable. These models reduce spatial characteristics which are critical in semantic segmentation. Thus instead of having pooling and fully connected layers, we can set up a convolution layer having a stride of 1 and the same padding. This preserves the input dimension and spatial information. However, this approach adds another disadvantage to the performance and cost - High memory and computation requirements.

To ease that problem, an encoder-Decoder architecture is introduced for semantic segmentation tasks. This network usually has 3 main components - Convolutions, down sampling and up sampling. On the encoder side, the network performs down sampling to perform deeper convolutions without requiring more memory. This part looks like a regular DCNN without fully connected layers. We can also use pre-trained models to extract features on the encoder side. Down sampling in neural networks can be done by using convolutional striding or pooling. The output of the first stage is a compressed feature vector with smaller spatial dimensions. Then we feed this compressed feature vector to the up sampling stage to reconstruct our original size. The goal is to increase the spatial dimensions so that the output is the same size as the original image. Here we use transpose convolutions to convert deep and narrow vectors to wider and shallow ones. Some of the popular networks implemented based on encoder-decoder architectures are FCNs, U-Net, SegNet etc. Experiments prove that the encoder-decoder architecture has achieved a good performance in many segmentation datasets.

## Results

| Models       | mIoU         | 
| ------------- |:-------------:| 
| U-Net      | 0.428| 
| FCN     | 0.422      |   
| DeepLabv3+ | 0.441      |    



Markdown | Less | Pretty
--- | --- | ---
*Still* | `renders` | **nicely**
1 | 2 | 3


