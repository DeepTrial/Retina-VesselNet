**[NOTE]** Since this project has upgraded to Tensorflow 2.3 on **18th March 2021**, you can find old branches which have stopped maintenance from:

- [2019-6-9] keras-tensorflow branch: https://github.com/DeepTrial/Retina-VesselNet/tree/keras-tensorflow-1.X
- [2018-5-2] keras-theano branch: https://github.com/DeepTrial/Retina-VesselNet/tree/keras-theano


# VesselNet
A Simple U-net model for Retinal Blood Vessel Segmentation with DRIVE dataset

![TestResult](https://i.imgur.com/pPMANyZ.jpg)


## Project Structure
We provide 2 version of projects: [jupyter notebook][1] and [`.py file`][3]. The implementation of these two versions is completely consistent. Choose one version and enjoy it!

### First to run
For the first time, I recommand to use the version of [jupyter notebook][1], it will give you an intuitive presentation. Different notebooks are made for different purpose:
- `EntireBookForColab.ipynb` contains complete part of projects such as process, train, test. Furthermore, it can be run on Google Colab
- `PreprocessIllustartion.ipynb` shows some preprocess methods for retinal images.
- `TestAndEvaluation.ipynb` is the part for evaluating and testing the model.
- `Training.ipynb` is the part for defining and training the model.

**Remenber to modify the dataset path according to your setting.**

### Pretrained Model
- Dataset can be found [here][2]. For Chinese, you can download [here][6].
- Pretrained model is coming soon...

### Train/Test your own image
If you want to test your own image, put your image to the the relevant dir and adjust the `patch_size`,`stride` according to your image size. 

## Citation
This project has been used in:
```
@inproceedings{2020Eye3DVas,
  title={Eye3DVas: three-dimensional reconstruction of retinal vascular structures by integrating fundus image features},
  author={ Yao Z. and  He K. and Zhou H. and Zhang Z. and Xing C. and Zhou F.},
  booktitle={Frontiers in Optics},
  year={2020},
}
```


## Reference
This project is based on the following 2 papers:


[U-Net: Convolutional Networks for Biomedical Image Segmentation](8)

[Densely Connected Convolutional Networks](7)



[1]: https://github.com/DeepTrial/Retina-VesselNet/tree/master/jupyter-notebook
[2]: https://drive.google.com/file/d/1RALItn7a-XIe-ebsghk6HL-T0btJI9w7/view?usp=sharing
[3]: https://github.com/DeepTrial/Retina-VesselNet/tree/master/py-network
[5]: https://github.com/orobix/retina-unet
[6]: https://pan.baidu.com/s/1EnKeNTGimzVRa9QedWjxlg
[7]: https://arxiv.org/pdf/1608.06993.pdf
[8]: https://arxiv.org/pdf/1505.04597.pdf



