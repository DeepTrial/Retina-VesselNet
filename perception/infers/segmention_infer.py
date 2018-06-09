# -- coding: utf-8 --
"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/11
"""
import glob,cv2,numpy as np
import matplotlib.pyplot as plt
from keras.models import model_from_json
from perception.bases.infer_base import InferBase
from configs.utils.img_utils import get_test_patches,pred_to_patches,recompone_overlap
from configs.utils.utils import visualize,gray2binary


class SegmentionInfer(InferBase):
	def __init__(self,config):
		super(SegmentionInfer, self).__init__(config)
		self.load_model()

	def load_model(self):
		self.model = model_from_json(open(self.config.hdf5_path+self.config.exp_name + '_architecture.json').read())
		self.model.load_weights(self.config.hdf5_path+self.config.exp_name+ '_best_weights.h5')

	def analyze_name(self,path):
		return (path.split('\\')[-1]).split(".")[0]

	def predict(self):
		predList=glob.glob(self.config.test_img_path+"*."+self.config.test_datatype)
		for path in predList:
			orgImg_temp=plt.imread(path)
			orgImg=orgImg_temp[:,:,1]*0.75+orgImg_temp[:,:,0]*0.25
			print("[Info] Analyze filename...",self.analyze_name(path))
			height,width=orgImg.shape[:2]
			orgImg = np.reshape(orgImg, (height,width,1))
			patches_pred,new_height,new_width,adjustImg=get_test_patches(orgImg,self.config)

			predictions = self.model.predict(patches_pred, batch_size=32, verbose=1)
			pred_patches=pred_to_patches(predictions,self.config)

			pred_imgs=recompone_overlap(pred_patches,self.config,new_height,new_width)
			pred_imgs=pred_imgs[:,0:height,0:width,:]

			adjustImg=adjustImg[0,0:height,0:width,:]
			print(adjustImg.shape)
			probResult=pred_imgs[0,:,:,0]
			binaryResult=gray2binary(probResult)
			resultMerge=visualize([adjustImg,binaryResult],[1,2])

			resultMerge=cv2.cvtColor(resultMerge,cv2.COLOR_RGB2BGR)

			cv2.imwrite(self.config.test_result_path+self.analyze_name(path)+"_merge.jpg",resultMerge)
			cv2.imwrite(self.config.test_result_path + self.analyze_name(path) + "_prob.bmp", (probResult*255).astype(np.uint8))


