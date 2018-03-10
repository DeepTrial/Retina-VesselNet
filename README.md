# VesselNet
A DenseBlock-Unet for Retinal Blood Vessel Segmentation

![TestResult](/DataSet/test.jpg)

## Model Instruction
This model is inspired by DenseNet and [@orobix/retina-unet][5],I modify the Conv2d block to DenseBlock and finally I get better result.The DenseBlock struct is shown below.This struct maximisely use the extracted feature.If u want further information,please read the [DenseNet Paper][3] and [code][4]

![DenseBlock](DenseBlock.png)


## Mertic
With data argumentation(only randomly rotate) and DenseBlock-Unet
Results on DRIVE database:

|Methods|AUC ROC on DRIVE|
|-:|-:|
|Liskowski|0.9790|
|Retina-Unet|0.9790|
|VesselNet|0.9793|

## Setup Model
to run this model,u need to setup basic python deep learning environment,please ensure that u have already installed CUDA and CUDNN.Python(i recommend to install anaconda) package and version requiremented:
requirements:
- python3+
- keras2.0+
- theano0.9+
- opencv3+
- matplotlib

please dowload the following files(including DRIVE database) if u want to train the model by yourself:

- Download [training data][1] and Extract to DataSet folder

the pretrained model is below:

- Download [pretrain model][2] and Extract to DataSet folder


## Test your own image
if u want to test your own image,put ur image to **TestFolder/origin**,and change the img_type of predict settings in **configurations.txt**,run xPredict.py to get your result.The result is in **TestFolder/result**


[1]: https://drive.google.com/open?id=1fDlnqeuGlX93lUkXEEWcImSFoIufLhRe
[2]: https://drive.google.com/open?id=1VVQveyxHIB4OT74Lk4M86GLddSupIZKO
[3]: https://arxiv.org/pdf/1608.06993.pdf
[4]: https://github.com/liuzhuang13/DenseNet 
[5]: https://github.com/orobix/retina-unet
[4]: https://github.com/liuzhuang13/DenseNet 
