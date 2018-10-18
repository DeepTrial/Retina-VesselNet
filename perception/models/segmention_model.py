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

	def build_model(self):
		inputs = Input((self.patch_height, self.patch_width,1))
		conv1 = Conv2D(32, (3, 3), padding='same')(inputs)  # 'valid'
		conv1 = LeakyReLU(alpha=0.3)(conv1)
		conv1 = Dropout(0.2)(conv1)
		conv1 = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,beta_initializer='RandomNormal', gamma_initializer='one')(conv1)
		conv1 = Conv2D(32, (3, 3), dilation_rate=2, padding='same')(conv1)
		conv1 = LeakyReLU(alpha=0.3)(conv1)
		conv1 = Conv2D(32, (3, 3), dilation_rate=4, padding='same')(conv1)
		conv1 = LeakyReLU(alpha=0.3)(conv1)
		pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

		# pool1 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(pool1)
		conv2 = Conv2D(64, (3, 3), padding='same')(pool1)  # ,activation='relu', padding='same')(pool1)
		conv2 = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                         beta_initializer='RandomNormal', gamma_initializer='one')(conv2)
		conv2 = LeakyReLU(alpha=0.3)(conv2)
		conv2 = Dropout(0.2)(conv2)
		conv2 = Conv2D(64, (3, 3), dilation_rate=2, padding='same')(conv2)
		conv2 = LeakyReLU(alpha=0.3)(conv2)
		conv2 = Conv2D(64, (3, 3), dilation_rate=4, padding='same')(
			conv2)  # ,W_regularizer=l2(0.01), b_regularizer=l2(0.01))(conv2)
		conv2 = LeakyReLU(alpha=0.3)(conv2)
		pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

		# crop = Cropping2D(cropping=((int(3 * patch_height / 8), int(3 * patch_height / 8)), (int(3 * patch_width / 8), int(3 * patch_width / 8))))(conv1)
		# conv3 = concatenate([crop,pool2], axis=1)
		conv3 = Conv2D(128, (3, 3), padding='same')(pool2)  # , activation='relu', padding='same')(conv3)
		conv3 = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                         beta_initializer='RandomNormal', gamma_initializer='one')(conv3)
		conv3 = LeakyReLU(alpha=0.3)(conv3)
		conv3 = Dropout(0.2)(conv3)
		conv3 = Conv2D(128, (3, 3), dilation_rate=2, padding='same')(
			conv3)  # ,W_regularizer=l2(0.01), b_regularizer=l2(0.01))(conv3)
		conv3 = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                         beta_initializer='RandomNormal', gamma_initializer='one')(conv3)
		conv3 = LeakyReLU(alpha=0.3)(conv3)

		conv3 = Conv2D(128, (3, 3), dilation_rate=4, padding='same')(conv3)
		conv3 = normalization.BatchNormalization(epsilon=2e-05, axis=3, momentum=0.9, weights=None,
		                                         beta_initializer='RandomNormal', gamma_initializer='one')(conv3)
		conv3 = LeakyReLU(alpha=0.3)(conv3)

		# up1 = UpSampling2D(size=(2, 2))(conv3)
		up1 = concatenate([UpSampling2D(size=(2, 2))(conv3), conv2], axis=3)
		conv4 = Conv2D(64, (3, 3), padding='same')(up1)
		conv4 = LeakyReLU(alpha=0.3)(conv4)
		conv4 = Dropout(0.2)(conv4)
		conv4 = Conv2D(64, (3, 3), padding='same')(conv4)
		conv4 = LeakyReLU(alpha=0.3)(conv4)
		# conv4 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv4)
		#
		# up2 = UpSampling2D(size=(2, 2))(conv4)
		up2 = concatenate([UpSampling2D(size=(2, 2))(conv4), conv1], axis=3)
		conv5 = Conv2D(32, (3, 3), padding='same')(up2)
		conv5 = LeakyReLU(alpha=0.3)(conv5)
		conv5 = Dropout(0.2)(conv5)
		conv5 = Conv2D(32, (3, 3), padding='same')(conv5)
		conv5 = LeakyReLU(alpha=0.3)(conv5)

		conv6 = Conv2D(self.num_seg_class + 1, (1, 1), padding='same')(conv5)
		conv6 = LeakyReLU(alpha=0.3)(conv6)
		# conv6 = normalization.BatchNormalization(epsilon=1e-06, mode=1, axis=-1, momentum=0.9, weights=None, beta_init='zero', gamma_init='one')(conv6)

		# for tensorflow
		# conv6 = core.Reshape((patch_height*patch_width,num_lesion_class+1))(conv6)
		# for theano
		conv6 = core.Reshape((self.patch_height * self.patch_width,self.num_seg_class + 1))(conv6)
		#conv6 = core.Permute((2, 1))(conv6)
		############
		act = Activation('softmax')(conv6)

		model = Model(inputs=inputs, outputs=act)
		model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['categorical_accuracy'])
		plot_model(model, to_file=os.path.join(self.config.checkpoint, "model.png"), show_shapes=True)
		self.model=model