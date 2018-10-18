"""
Copyright (c) 2018. All rights reserved.
Created by Resnick Xing on 2018/5/11
"""

import keras,os
from keras.models import Model
from keras.layers.merge import add,multiply
from keras.layers import Lambda,Input, Conv2D,Conv2DTranspose,Conv2DTranspose, MaxPooling2D, UpSampling2D,Cropping2D, core, Dropout,normalization,concatenate,Activation
from keras import backend as K
from keras.layers.core import Layer, InputSpec
from keras.layers.advanced_activations import LeakyReLU
from keras.utils import plot_model

from perception.bases.model_base import ModelBase

class SegmentionModel(ModelBase):
	def __init__(self,config=None):
		super(SegmentionModel, self).__init__(config)

		self.patch_height=config.patch_height
		self.patch_width = config.patch_width
		self.num_seg_class=config.seg_num

		self.build_model()
		self.save()

	def DenseBlock(self,inputs, outdim):

		inputshape = K.int_shape(inputs)
		bn = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                      beta_initializer='zero', gamma_initializer='one')(inputs)
		act = Activation('relu')(bn)
		conv1 = Conv2D(outdim, (3, 3), activation=None, padding='same')(act)

		if inputshape[3] != outdim:
			shortcut = Conv2D(outdim, (1, 1), padding='same')(inputs)
		else:
			shortcut = inputs
		result1 = add([conv1, shortcut])

		bn = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                      beta_initializer='zero', gamma_initializer='one')(result1)
		act = Activation('relu')(bn)
		conv2 = Conv2D(outdim, (3, 3), activation=None, padding='same')(act)
		result = add([result1, conv2, shortcut])
		result = Activation('relu')(result)
		return result

	def build_model(self):
		inputs = Input((self.patch_height, self.patch_width, 1))
		conv1 = Conv2D(32, (1, 1), activation=None, padding='same')(inputs)
		conv1 = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                         beta_initializer='zero', gamma_initializer='one')(conv1)
		conv1 = Activation('relu')(conv1)

		conv1 = self.DenseBlock(conv1, 32)  # 48
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		conv2 = self.DenseBlock(pool1, 64)  # 24
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		conv3 = self.DenseBlock(pool2, 64)  # 12
		pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

		conv4 = self.DenseBlock(pool3, 64)  # 12

		up1 = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(conv4)
		up1 = concatenate([up1, conv3], axis=3)

		conv5 = self.DenseBlock(up1, 64)

		up2 = Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(conv5)
		up2 = concatenate([up2, conv2], axis=3)

		conv6 = self.DenseBlock(up2, 64)

		up3 = Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(conv6)
		up3 = concatenate([up3, conv1], axis=3)

		conv7 = self.DenseBlock(up3, 32)

		conv8 = Conv2D(self.num_seg_class + 1, (1, 1), activation='relu', padding='same')(conv7)
		# conv6 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv6)

		# for tensorflow
		conv8 = core.Reshape((self.patch_height*self.patch_width,self.num_seg_class + 1))(conv8)
		# for theano
		#conv8 = core.Reshape(((self.num_seg_class + 1), self.patch_height * self.patch_width))(conv8)
		#conv8 = core.Permute((2, 1))(conv8)
		############
		act = Activation('softmax')(conv8)

		model = Model(inputs=inputs, outputs=act)
		model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
		plot_model(model, to_file=os.path.join(self.config.checkpoint, "model.png"), show_shapes=True)
		self.model = model