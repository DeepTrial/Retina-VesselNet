"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/10
"""
import random,numpy as np,cv2
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback

from perception.bases.trainer_base import TrainerBase
from configs.utils.utils import genMasks,visualize
from configs.utils.img_utils import img_process

class SegmentionTrainer(TrainerBase):
	def __init__(self,model,data,config):
		super(SegmentionTrainer, self).__init__(model, data, config)
		self.model=model
		self.data=data
		self.config=config
		self.callbacks=[]
		self.init_callbacks()

	def init_callbacks(self):
		self.callbacks.append(
			ModelCheckpoint(
				filepath=self.config.hdf5_path+self.config.exp_name+ '_best_weights.h5',
		        verbose=1,
		        monitor='val_loss',
		        mode='auto',
		        save_best_only=True
			)
		)

		self.callbacks.append(
			TensorBoard(
				log_dir=self.config.checkpoint,
				write_images=True,
				write_graph=True,
			)
		)

	def train(self):
		gen=DataGenerator(self.data,self.config)
		gen.visual_patch()
		hist = self.model.fit_generator(gen.train_gen(),
		    epochs=self.config.epochs,
		    steps_per_epoch=self.config.subsample * self.config.total_train / self.config.batch_size,
		    verbose=1,
		    callbacks=self.callbacks,
		    validation_data=gen.val_gen(),
			validation_steps=int(self.config.subsample * self.config.total_val / self.config.batch_size)
		)
		self.model.save_weights(self.config.hdf5_path+self.config.exp_name+'_last_weights.h5', overwrite=True)



class DataGenerator():
	"""
	load image (Generator)
	"""
	def __init__(self,data,config):
		self.train_img=img_process(data[0])
		self.train_gt=data[1]/255.
		self.val_img=img_process(data[2])
		self.val_gt=data[3]/255.
		self.config=config

	def _CenterSampler(self,attnlist,class_weight,Nimgs):
		"""
		围绕目标区域采样
		:param attnlist:  目标区域坐标
		:return: 采样的坐标
		"""
		class_weight = class_weight / np.sum(class_weight)
		p = random.uniform(0, 1)
		psum = 0
		for i in range(class_weight.shape[0]):
			psum = psum + class_weight[i]
			if p < psum:
				label = i
				break
		if label == class_weight.shape[0] - 1:
			i_center = random.randint(0, Nimgs - 1)
			x_center = random.randint(0 + int(self.config.patch_width / 2), self.config.width - int(self.config.patch_width / 2))
			# print "x_center " +str(x_center)
			y_center = random.randint(0 + int(self.config.patch_height / 2), self.config.height - int(self.config.patch_height / 2))
		else:
			t = attnlist[label]
			cid = random.randint(0, t[0].shape[0] - 1)
			i_center = t[0][cid]
			y_center = t[1][cid] + random.randint(0 - int(self.config.patch_width / 2), 0 + int(self.config.patch_width / 2))
			x_center = t[2][cid] + random.randint(0 - int(self.config.patch_width / 2), 0 + int(self.config.patch_width / 2))

		if y_center < self.config.patch_width / 2:
			y_center = self.config.patch_width / 2
		elif y_center > self.config.height - self.config.patch_width / 2:
			y_center = self.config.height - self.config.patch_width / 2

		if x_center < self.config.patch_width / 2:
			x_center = self.config.patch_width / 2
		elif x_center > self.config.width - self.config.patch_width / 2:
			x_center = self.config.width - self.config.patch_width / 2

		return i_center, x_center, y_center

	def _genDef(self,train_imgs,train_masks,attnlist,class_weight):
		"""
		图片取块生成器模板
		:param train_imgs: 原始图
		:param train_masks:  原始图groundtruth
		:param attnlist:  目标区域list
		:return:  取出的训练样本
		"""
		while 1:
			Nimgs=train_imgs.shape[0]
			for t in range(int(self.config.subsample * self.config.total_train / self.config.batch_size)):
				X = np.zeros([self.config.batch_size,self.config.patch_height, self.config.patch_width,1])
				Y = np.zeros([self.config.batch_size, self.config.patch_height * self.config.patch_width, self.config.seg_num + 1])
				for j in range(self.config.batch_size):
					[i_center, x_center, y_center] = self._CenterSampler(attnlist,class_weight,Nimgs)
					patch = train_imgs[i_center, int(y_center - self.config.patch_height / 2):int(y_center + self.config.patch_height / 2),int(x_center - self.config.patch_width / 2):int(x_center + self.config.patch_width / 2),:]
					patch_mask = train_masks[i_center, :, int(y_center - self.config.patch_height / 2):int(y_center + self.config.patch_height / 2),int(x_center - self.config.patch_width / 2):int(x_center + self.config.patch_width / 2)]
					X[j, :, :, :] = patch
					Y[j, :, :] = genMasks(np.reshape(patch_mask, [1, self.config.seg_num, self.config.patch_height, self.config.patch_width]),self.config.seg_num)
				yield (X, Y)

	def train_gen(self):
		"""
		训练样本生成器
		"""
		class_weight=[1.0,0.0]
		attnlist=[np.where(self.train_gt[:,0,:,:]==np.max(self.train_gt[:,0,:,:]))]
		return self._genDef(self.train_img,self.train_gt,attnlist,class_weight)

	def val_gen(self):
		"""
		验证样本生成器
		"""
		class_weight = [1.0,0.0]
		attnlist = [np.where(self.val_gt[:, 0, :, :] == np.max(self.val_gt[:, 0, :, :]))]
		return self._genDef(self.val_img, self.val_gt, attnlist,class_weight)

	def visual_patch(self):
		gen=self.train_gen()
		(X,Y)=next(gen)
		image=[]
		mask=[]
		print("[INFO] Visualize Image Sample...")
		for index in range(self.config.batch_size):
			image.append(X[index])
			mask.append(np.reshape(Y,[self.config.batch_size,self.config.patch_height,self.config.patch_width,self.config.seg_num+1])[index,:,:,0])
		if self.config.batch_size%4==0:
			row=self.config.batch_size/4
			col=4
		else:
			if self.config.batch_size % 5!=0:
				row = self.config.batch_size // 5+1
			else:
				row = self.config.batch_size // 5
			col = 5
		imagePatch=visualize(image,[row,col])
		maskPatch=visualize(mask,[row,col])
		cv2.imwrite(self.config.checkpoint+"image_patch.jpg",imagePatch)
		cv2.imwrite(self.config.checkpoint + "groundtruth_patch.jpg", maskPatch)