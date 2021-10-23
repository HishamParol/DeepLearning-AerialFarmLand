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

The data needed for this project was provided by [Agricultural-Vision](https://www.agriculture-vision.com/agriculture-vision-2020/dataset)- an independent research
board for the research and development of computer vision technology for agriculture. The dataset is
large scale and high quality images of aerial farmlands for the advanced study of agricultural patterns.
The dataset was precisely annotated by professional agronomists with a strict quality assurance process.
The proposed agricultural vision contains 94,986 samples of images and the subset of this dataset
containing 12,901 samples was used for challenge competition. Same was used for this project as well.
The dataset contains nine types of annotations - double plant, dry down, endow, nutrient deficiency,
planter skip, storm damage, water, waterway and weed cluster. In this project only weed clusters are
used for the segmentation task. These images were captured by special mounted camera on Drones
flown over various corn and soybean fields around Illinois and Iowa (USA).
Colons can be used to align columns.

| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

There must be at least 3 dashes separating each header cell.
The outer pipes (|) are optional, and you don't need to make the 
raw Markdown line up prettily. You can also use inline Markdown.

Markdown | Less | Pretty
--- | --- | ---
*Still* | `renders` | **nicely**
1 | 2 | 3


| Tables        | Are           | Cool  |
| ------------- |:-------------:| -----:|
| col 3 is      | right-aligned | $1600 |
| col 2 is      | centered      |   $12 |
| zebra stripes | are neat      |    $1 |

There must be at least 3 dashes separating each header cell.
The outer pipes (|) are optional, and you don't need to make the 
raw Markdown line up prettily. You can also use inline Markdown.

Markdown | Less | Pretty
--- | --- | ---
*Still* | `renders` | **nicely**
1 | 2 | 3
