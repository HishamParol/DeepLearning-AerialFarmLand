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

### 1. Data Pre-processing

#### Choosing images with weed clusters

In this process, Masked Images in the ‘weed_cluster’ folder were analysed to study the presence of weeds. After careful analysis, it is found that the images that don't have weed clusters are fully background images.The black area represents the background (crops) and white represents the annotated weed clusters.Converting images to NumPy arrays, and analysing the pixel information, all the pixel values in the black images are of value 255. Thus a python code is implemented to select only images having different pixel values (black and white pixels) in their corresponding NumPy arrays.

#### Geometric Transformation - Data Augmentation
Choice of geometric transformation is primarily based on understanding the augmentation in the context of the safety of the application. Safety refers to the likelihood of preserving the label post transform. For example, cropping, rotation, flipping are safe on agricultural field datasets. These augmentations address the broad scope of challenges in the cultivation of corn and soybeans. This data augmentation technique is used for improving the quality of the dataset with real-world weeds captured by drones. Key assumptions include capturing aerial images at various angles, depths and directions.

#### Colour Transformation - Data Augmentation
Performing transformations in the colour channels space is another approach that is very reasonable to achieve. Digitally, Image data is encoded as a tensor of the dimension (height × width × colour channels). Colour augmentation includes isolating a single colour channel such as R, G, B. Histogram equalization for adjusting the contrast, Controlling the Brightness, Saturation and Hue. These techniques address various environmental conditions in the real world such as sunny weather (High brightness), cloud shades, nature of soils etc. Colour augmentation is used to stimulate a brighter or darker environment by decreasing or increasing the pixel values respectively.

### 2. Build and Train models
In this project, the dataset is trained on prominent and widely used deep learning-based semantic segmentation models such as FCN, U-Net and DeepLabv3+. Pre-trained models such as VGG16 and Xception are used in the encoder side of FCN and DeepLabv3+.

#### U-Net Model
As the name suggests. U-Net has a U -Shaped architecture, where the first part is a contraction path and is followed by an expansion path. U-Net utilises dense skip connections to fully leverage the features from each layer. The contraction path is also called the encoder side and the expansion path is called the decoder. The characteristics from each layer in the contraction path are connected to the symmetrical layers in the expansion path. The network architecture is shown in the figure below.
Read more about U-net here @

#### Fully Convolutional Network (FCN) Model
The first model employed here is Fully Convolutional Network (FCN). They employ locally connected layers such as convolution, pooling and up sampling. In a regular CNN network, if the fully connected layers at the end are replaced by a convolutional layer. We get coarse spatial features as output instead of vectors. This can be further up sampled to get the segmented image.

#### DeepLabv3+ - DeepLab series Model
Invented by Google and the most advanced model of the DeepLab series, DeepLabv3+ extends DeepLabv3 by combining a simple yet efficient decoder module to improve the segmentation results, especially along object boundaries. The previous version - DeepLabv3 employs several parallel atrous convolutions with different rates to capture the contextual information at multiple scales. The encoder-decoder architecture used in U-net has been proved to be useful in recovering spatial information. Both of these concepts are combined In DeepLabv3+ making it a superior model in the DeepLab series. The DeepLab v3+ offers an architecture for controlling signal decimation and learning multi-scale
contextual features. The main three components of DeeplabV3+ are Atrous convolution and Atrous Spatial Pyramid Pooling
(ASPP), Encoder and Decoder

Read more about DeepLabv3+ here

#### Deep Learning framework and library
The architecture is built using Keras - A high-level API for Google's TensorFlow library. Keras is a popular choice for deep learning as it is highly integrated into TensorFlow. TensorFlow also offers various tools for visualization, debugging, running models on browsers etc. Building a model in Keras is straightforward and can be implemented using fewer lines of code unlike other libraries such as PyTorch. Keras is ideal for building complex and advanced models using its functional API and layers.The Keras architecture of the proposed three models is illustrated in the diagram below.

### 3. Transfer Learning
Transfer learning is a method, where a model used for a particular task is reused to train a second model. Given the computation and time resources of deep learning models, Transfer learning is used as a starting point to allow rapid progress and improve performance.Assuming that the general features of the base model may align with the features of weed such as shape, size and colour, pre-trained models trained on PASCAL VOC 2011 and Cityscaper Image datasets were chosen.

In this work, the following backbone models were used as the base model:
###### VGG16: Used in FCN Network ‘Very Deep Convolutional Networks for Large-Scale Image Recognition” Pretrained on PASCAL VOC 2011 dataset
###### Xception: Used in DeepLabV3+ pre-trained on PASCAL VOC 2011 and City Scrapes
###### MobileNetV2: Used in DeepLabV3+ pre-trained on PASCAL VOC 2011 and City Scrapes.

### 4. Evaluation results
#### Evaluation Metrics
In this project, mean Intersection-over-Union (mIoU) is used as a main quantitative evaluation metric, which is one of the most commonly used measures in semantic segmentation datasets. The mIoU is computed as:

Where c is the number of annotation types (c = 2 in this dataset, with 1 pattern + background), P(c) and T(c) are the predicted mask and ground truth mask of class c respectively. For pixels in each predicted image, a prediction of a masked area label will be counted as a correct pixel classification for that label, and a prediction that does not contain any ground truth labels will be counted as an incorrect classification for all ground truth labels.

#### Hyperparameter Tuning
It is important to say that at this level little is known about the exact impact of hyper-parameters on the performance of models.The strategy of this stage is based on a trial and error approach, where we customize hyperparameters, expecting to see improvement in results. Following parameters have been selected for this process -  batch size, selection of optimizers, loss functions, learning rates and evaluation metrics.

##### Batch Size
Number of samples processed before the model gets updated is known as batch size. A large batch size requires more computation power and training becomes slower. On the other hand, using a small batch size cannot guarantee global minima but converges to a good solution.

| Models       | Batch Size         | mIoU|
| ------------- |:-------------:|:-------------:| 
| FCN     |8  | 0.428| 
| FCN     |16  | 0.422      |   
| U-Net |8 | 0.441      | 
| U-Net      |16  | 0.428| 
| DeepLabv3+ |8  | 0.422      |   
| DeepLabv3+ |16 | 0.441      | 

##### Selection of optimizers
##### Loss functions
##### Learning rate



### 5. Prediction 



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


