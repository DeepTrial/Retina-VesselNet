**[NOTE]** Since this project has upgraded to Tensorflow 2.3 on **18th March 2021**, you can find old branch from:

- [2019-6-9] keras-tensorflow branch: https://github.com/DeepTrial/Retina-VesselNet/tree/keras-tensorflow-1.X
- [2018-5-2] keras-theano branch: https://github.com/DeepTrial/Retina-VesselNet/tree/keras-theano


# VesselNet
A Simple U-net model for Retinal Blood Vessel Segmentation with DRIVE dataset

![TestResult](https://i.imgur.com/pPMANyZ.jpg)



## Project Structure
The structure is based on my own [DL_Segmention_Template][1]. Difference between this project and the template is that we have metric module in dir: `perception/metric/`. To get more Information about the structure please see readme in [DL_Segmention_Template][1].

You can find model parameter in **configs/segmention_config.json**.

### First to run
**please run main_trainer.py first time**, then you will get data_route in experiment dir. Put your data in there, now you can run `main_trainer.py` again to train a model. 

### Pretrained Model
The model is trained with *DRIVE dataset* on my own desktop (intel i7-7700hq,24g,gtx1050 2g) within 30 minutes.
Dataset and pretrained model can be found [here][2]. For Chinese, you can download [here][6].

### Test your own image
If u want to test your own image, put ur image to **(VesselNet)/test/origin**, and change the img_type of predict settings in **configs/segmention_config.json**, run `main_test.py` to get your result. The result is in **(VesselNet)/test/result**

## Reference
This project is based on the following 2 papers:



[U-Net: Convolutional Networks for Biomedical Image Segmentation](8)

[Densely Connected Convolutional Networks](7)



[1]: https://github.com/DeepTrial/DL_Segmention_Template
[2]: https://drive.google.com/file/d/1RALItn7a-XIe-ebsghk6HL-T0btJI9w7/view?usp=sharing
[3]: https://arxiv.org/pdf/1608.06993.pdf
[4]: https://github.com/liuzhuang13/DenseNet 
[5]: https://github.com/orobix/retina-unet
[6]: https://pan.baidu.com/s/1EnKeNTGimzVRa9QedWjxlg
[7]: https://arxiv.org/pdf/1608.06993.pdf
[8]: https://arxiv.org/pdf/1505.04597.pdf



